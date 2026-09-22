"""
Long-context evaluation module for ApexDecision External Benchmark Suite.
Evaluates hierarchical context scaling across 32K, 64K, 128K, 256K, 512K, and 1,000,000 tokens.
Measures:
- Needle-in-haystack retrieval accuracy
- Latency scaling curve (ms)
- Peak GPU memory allocation (MB)
- Accuracy degradation vs short context baseline
"""

import time
import torch
from typing import Dict, Any, List, Optional

from apex_decision.context.long_context.hierarchical_engine import HierarchicalLongContextEngine
from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine


class LongContextEvaluator:
    """Evaluates context lengths from 32K to 1M tokens."""

    def __init__(self, engine: Optional[ApexTypedDecisionEngine] = None):
        self.engine = engine or ApexTypedDecisionEngine()
        self.context_engine = HierarchicalLongContextEngine()

    def run_long_context_eval(self, depths: Optional[List[str]] = None) -> Dict[str, Any]:
        target_depths = depths or ["32K", "64K", "128K", "256K", "512K", "1M"]
        token_counts = {
            "32K": 32_000,
            "64K": 64_000,
            "128K": 128_000,
            "256K": 256_000,
            "512K": 512_000,
            "1M": 1_000_000
        }

        # Background distractor sentence
        filler = (
            "Standard telemetry confirms network throughput remaining within nominal variance. "
            "All auxiliary subsystems report green across primary operating registers. "
            "Routine cache eviction completed successfully at scheduled maintenance window. "
        )

        results_by_depth = {}
        baseline_acc = 100.0

        for depth_label in target_depths:
            tokens_target = token_counts.get(depth_label, 32_000)
            words_needed = int(tokens_target * 0.75)
            filler_words = len(filler.split())
            repeats = max(1, words_needed // filler_words)

            # Generate haystack
            part1 = (filler * (repeats // 2))
            part2 = (filler * (repeats - (repeats // 2)))

            # Insert critical needle fact
            needle = "CRITICAL FACT: The master cryptographic encryption key is SIGMA-9942."
            synthetic_context = f"{part1}\n\n{needle}\n\n{part2}"

            query = "Is the master cryptographic encryption key SIGMA-9942?"

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            # Process via hierarchical context engine
            ctx_res = self.context_engine.process_long_context(synthetic_context, query)
            retrieved_snippet = ctx_res["selected_evidence"]

            # Perform decision with student engine
            dec_res = self.engine.predict_noul(question=query, state=retrieved_snippet)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            peak_vram_mb = 0.0
            if torch.cuda.is_available():
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Verification: needle must be in retrieved snippet and decision must be SUPPORTED
            retrieval_ok = "SIGMA-9942" in retrieved_snippet
            decision_ok = (dec_res.verdict == "SUPPORTED")
            accuracy = 100.0 if (retrieval_ok and decision_ok) else 0.0

            results_by_depth[depth_label] = {
                "depth_tokens": tokens_target,
                "retrieval_success": retrieval_ok,
                "decision_verdict": dec_res.verdict,
                "accuracy": accuracy,
                "latency_ms": round(elapsed_ms, 2),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "compression_ratio": ctx_res.get("compression_ratio", "N/A"),
                "degradation_vs_short_context": round(baseline_acc - accuracy, 2)
            }

        avg_acc = sum(v["accuracy"] for v in results_by_depth.values()) / max(1, len(results_by_depth))

        return {
            "benchmark_id": "long_context_scaling",
            "sample_count": len(results_by_depth),
            "retrieval_accuracy": round(avg_acc, 2),
            "tested_depths": list(results_by_depth.keys()),
            "depth_details": results_by_depth
        }
