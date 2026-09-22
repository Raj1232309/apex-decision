"""
Performance and hardware profiling module for ApexDecision External Benchmark Suite.
Measures:
- Cold-start latency (initial forward pass)
- Warm latency distribution (P50, P90, P95, P99, mean)
- System throughput (inferences / second)
- Peak GPU memory allocation (MB)
- Peak system RAM consumption (MB)
- Latency scaling curve across sequence length
"""

import time
import psutil
import numpy as np
import torch
from typing import Dict, Any, List, Optional

from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine


class PerformanceEvaluator:
    """Profiles computational throughput, latency, and memory footprint."""

    def __init__(self, engine: Optional[ApexTypedDecisionEngine] = None):
        self.engine = engine or ApexTypedDecisionEngine()

    def run_performance_eval(self, warmup_runs: int = 5, test_runs: int = 30) -> Dict[str, Any]:
        state = "Standard telemetry confirms throughput nominal. Cluster health operational across all zones."
        query = "Is cluster health operational across all zones?"

        process = psutil.Process()
        ram_before_mb = process.memory_info().rss / (1024 * 1024)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # 1. Measure Cold Start
        t_cold_0 = time.perf_counter()
        _ = self.engine.predict_noul(question=query, state=state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cold_start_ms = (time.perf_counter() - t_cold_0) * 1000.0

        # 2. Warmup
        for _ in range(warmup_runs):
            _ = self.engine.predict_noul(question=query, state=state)

        # 3. Timed Warm Iterations
        latencies = []
        for _ in range(test_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = self.engine.predict_noul(question=query, state=state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

        # 4. Memory Profiling
        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        ram_after_mb = process.memory_info().rss / (1024 * 1024)

        # 5. Throughput
        total_time_sec = sum(latencies) / 1000.0
        throughput = test_runs / max(1e-6, total_time_sec)

        # 6. Sequence Length Scaling Curve
        seq_scaling = {}
        for seq_words in [64, 128, 256, 512, 1024]:
            scaled_state = (state + " ") * (seq_words // len(state.split()))
            t_s0 = time.perf_counter()
            _ = self.engine.predict_noul(question=query, state=scaled_state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seq_scaling[f"seq_{seq_words}_words"] = round((time.perf_counter() - t_s0) * 1000.0, 2)

        lat_arr = np.array(latencies)
        return {
            "benchmark_id": "performance_latency_profiling",
            "sample_count": test_runs,
            "cold_start_ms": round(cold_start_ms, 2),
            "warm_latency_ms": {
                "p50": round(float(np.percentile(lat_arr, 50)), 2),
                "p90": round(float(np.percentile(lat_arr, 90)), 2),
                "p95": round(float(np.percentile(lat_arr, 95)), 2),
                "p99": round(float(np.percentile(lat_arr, 99)), 2),
                "mean": round(float(np.mean(lat_arr)), 2),
                "min": round(float(np.min(lat_arr)), 2),
                "max": round(float(np.max(lat_arr)), 2)
            },
            "throughput_req_per_sec": round(throughput, 2),
            "peak_gpu_vram_mb": round(peak_vram_mb, 2),
            "peak_system_ram_mb": round(ram_after_mb, 2),
            "sequence_length_scaling_ms": seq_scaling
        }
