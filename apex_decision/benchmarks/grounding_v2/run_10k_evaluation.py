"""
Evaluation runner for ApexDecision v2 on the Frozen 10k Benchmark.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apex_decision.benchmarks.grounding_v2.entailment_verifier import ApexEntailmentVerifierV2
from apex_decision.benchmarks.grounding_v2.evaluate_v2 import evaluate_grounding_v2


def run_evaluation(
    benchmark_path: Optional[str] = None,
    output_results_path: Optional[str] = None,
    output_predictions_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    temperature: float = 1.0
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../.."))
    lora_dir = os.path.join(project_root, "apex_decision", "build", "grounding_lora")
    if not os.path.exists(lora_dir):
        lora_dir = os.path.join(project_root, "build", "grounding_lora")

    benchmark_path = benchmark_path or os.path.join(base_dir, "test_frozen_10k.json")
    output_results_path = output_results_path or os.path.join(base_dir, "final_10k_results.json")
    output_predictions_path = output_predictions_path or os.path.join(base_dir, "final_10k_predictions.json")

    print(f"Initializing ApexEntailmentVerifierV2...")
    print(f"LoRA directory: {lora_dir}")
    print(f"Benchmark path: {benchmark_path}")
    print(f"Max samples: {max_samples if max_samples else 'All 10,000'}")

    verifier = ApexEntailmentVerifierV2(
        base_model_name="answerdotai/ModernBERT-base",
        lora_dir=lora_dir,
        use_structured_checks=True,
        use_decomposition=True,
        temperature=temperature
    )

    results, predictions = evaluate_grounding_v2(
        predict_fn=verifier.verify,
        dataset_path=benchmark_path,
        model_name="ApexDecision_v2_ModernBERT_LoRA_MultiStage",
        max_samples=max_samples
    )

    # Save results
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {output_results_path}")

    with open(output_predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output_predictions_path}")

    return results, predictions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()

    run_evaluation(benchmark_path=args.benchmark, max_samples=args.max_samples)
