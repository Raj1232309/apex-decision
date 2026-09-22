"""
ApexDecision V7.4: Architecture & Parameter Scaling Benchmark.
Systematically benchmarks candidate architectures across the authorized spectrum
(24M, 53M, 105M, 151M, 207M, 245M) on NVIDIA GeForce RTX 2050 (4GB VRAM)
for parameter count, weights memory, peak inference VRAM, single-query latency,
and throughput against Laya (ModernBERT-large 421M).
"""

import os
import sys
import time
import json
import torch
import numpy as np
from typing import Dict, Any, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "laya_repo"))

from apex_decision.models.apex_student.config import ApexStudentConfig
from apex_decision.models.apex_student.model import ApexStudentModel
from apex_decision.models.apex_student.tokenizer import ApexTokenizer
from apex_decision.inference.apex_student.runtime import ApexStudentAgent


SCALE_CONFIGS = [
    {"name": "Baseline-24M", "hidden": 384, "layers": 8, "heads": 8, "inter": 1536},
    {"name": "Scale-50M",    "hidden": 512, "layers": 12, "heads": 8, "inter": 2048},
    {"name": "Scale-100M",   "hidden": 768, "layers": 11, "heads": 12, "inter": 3072},
    {"name": "Scale-150M",   "hidden": 896, "layers": 12, "heads": 14, "inter": 3584},
    {"name": "Scale-200M",   "hidden": 1024, "layers": 13, "heads": 16, "inter": 4096},
    {"name": "Scale-245M",   "hidden": 1024, "layers": 16, "heads": 16, "inter": 4096},
]


def benchmark_all_scales():
    print("=" * 85)
    print("APEXDECISION V7.4: PARAMETER SCALE BENCHMARK (RTX 2050 4GB)")
    print("Evaluating models under the 250M hard upper limit ceiling")
    print("=" * 85)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok_dir = "apex_decision/models/apex_student/tokenizer_weights"
    tok = ApexTokenizer.from_pretrained(tok_dir)

    results = []

    test_state = (
        "Enterprise quarterly report: Total recurring revenue rose by 14.2% year-over-year "
        "to $412.5M, driven by strong cloud adoption and low customer churn. Capital expenditures "
        "totaled $45.0M, while operating margins expanded by 220 basis points."
    )
    test_q_noul = {"q": {"type": "noul", "instructions": "Did total revenue exceed $400.0M?"}}
    test_q_choice = {
        "q": {
            "type": "choice",
            "instructions": "Classify the performance trend.",
            "criteria": {"growth": "Positive revenue expansion", "decline": "Contraction", "flat": "No change"}
        }
    }

    for spec in SCALE_CONFIGS:
        name = spec["name"]
        h = spec["hidden"]
        l = spec["layers"]
        a = spec["heads"]
        inter = spec["inter"]

        cfg = ApexStudentConfig(
            vocab_size=max(4096, tok.vocab_size),
            hidden_size=h,
            num_hidden_layers=l,
            num_attention_heads=a,
            intermediate_size=inter
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = ApexStudentModel(cfg).to(dev).half() # fp16 for fast inference
        total_params = sum(p.numel() for p in model.parameters())
        weights_mb = (total_params * 2) / (1024 * 1024)

        agent = ApexStudentAgent(model, tok, cfg, device=dev)

        # Warmup
        for _ in range(5):
            _ = agent.predict(test_state, test_q_noul)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Latency benchmark: 50 passes
        timings = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = agent.predict(test_state, test_q_noul)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

        p50 = float(np.percentile(timings, 50))
        p95 = float(np.percentile(timings, 95))
        qps = 1000.0 / p50
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Verify Choice prediction
        choice_res = agent.predict(test_state, test_q_choice)
        choice_pred = choice_res["answers"]["q"].get("choice")

        results.append({
            "name": name,
            "hidden": h,
            "layers": l,
            "heads": a,
            "params_m": round(total_params / 1e6, 2),
            "weights_fp16_mb": round(weights_mb, 1),
            "peak_vram_mb": round(peak_vram, 1),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "throughput_qps": round(qps, 1),
            "practical_on_rtx2050": peak_vram < 3500
        })

        print(f"[{name:12s}] {total_params/1e6:6.2f}M params | Weights: {weights_mb:5.1f} MB | Peak VRAM: {peak_vram:6.1f} MB | Latency p50: {p50:5.2f} ms | QPS: {qps:5.1f}")

        del model
        del agent
        torch.cuda.empty_cache()

    # Benchmark Laya for reference
    import laya
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    laya_agent = laya.load("laya_weights", device=dev)
    for _ in range(3):
        _ = laya_agent.predict(test_state, test_q_noul)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    laya_timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = laya_agent.predict(test_state, test_q_noul)
        torch.cuda.synchronize()
        laya_timings.append((time.perf_counter() - t0) * 1000.0)

    laya_p50 = float(np.percentile(laya_timings, 50))
    laya_p95 = float(np.percentile(laya_timings, 95))
    laya_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    summary = {
        "device": "NVIDIA GeForce RTX 2050 (4GB)",
        "laya_reference": {
            "name": "Laya (ModernBERT-large)",
            "params_m": 421.0,
            "peak_vram_mb": round(laya_vram, 1),
            "latency_p50_ms": round(laya_p50, 2),
            "latency_p95_ms": round(laya_p95, 2),
            "throughput_qps": round(1000.0 / laya_p50, 1)
        },
        "student_scales": results
    }

    out_file = "apex_decision/reports/parameter_scaling_benchmark.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85)
    print("PARAMETER SCALING COMPARISON SCOREBOARD (RTX 2050 4GB)")
    print("=" * 85)
    print(f"{'Model Architecture':22s} | {'Params':>8s} | {'Weights':>10s} | {'Peak VRAM':>10s} | {'p50 Latency':>12s} | {'Speed vs Laya':>14s}")
    print("-" * 85)
    print(f"{'Laya (Teacher)':22s} | {'421.0M':>8s} | {'~840 MB':>10s} | {laya_vram:8.1f} MB | {laya_p50:10.2f} ms | {'1.0x (Baseline)':>14s}")
    for r in results:
        speedup = f"{laya_p50 / r['latency_p50_ms']:.2f}x Faster"
        p_str = f"{r['params_m']:.1f}M"
        w_str = f"{r['weights_fp16_mb']:.1f} MB"
        v_str = f"{r['peak_vram_mb']:.1f} MB"
        lat_str = f"{r['latency_p50_ms']:.2f} ms"
        print(f"{r['name']:22s} | {p_str:>8s} | {w_str:>10s} | {v_str:>10s} | {lat_str:>12s} | {speedup:>14s}")
    print("=" * 85)

    return summary


if __name__ == "__main__":
    benchmark_all_scales()
