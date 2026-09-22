"""
Standard Evaluation Harness for Grounding Verification Models on ApexDecision Benchmarks.
"""

import os
import json
import time
from typing import List, Dict, Any, Callable, Optional

from .dataset_schema import GroundingExample, GroundingPrediction, GroundingLabel
from .metrics import calculate_grounding_metrics
from .error_analysis import run_error_analysis


def load_dataset_split(split_path: str) -> List[GroundingExample]:
    with open(split_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [GroundingExample(**item) for item in raw_data]


def evaluate_predictor(
    predictor_fn: Callable[[str, str], Dict[str, Any]],
    dataset_path: str,
    output_results_path: Optional[str] = None,
    output_error_path: Optional[str] = None,
    model_name: str = "CandidateModel"
) -> Dict[str, Any]:
    """
    Evaluates a predictor function against a dataset split.
    predictor_fn takes (source, claim) and returns:
    {
      "decision": "SUPPORTED | NOT_SUPPORTED",
      "probability": float,
      "evidence_spans": list of str,
      "reason": str,
      "latency_us": int
    }
    """
    examples = load_dataset_split(dataset_path)
    print(f"Evaluating {model_name} on {len(examples)} examples from {dataset_path}...")

    predictions: List[GroundingPrediction] = []
    t_start = time.perf_counter()

    for ex in examples:
        t0 = time.perf_counter_ns()
        res = predictor_fn(ex.source, ex.claim)
        dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)

        latency_us = res.get("latency_us", dt_us)
        prob = float(res.get("probability", 0.5))
        dec = res.get("decision", GroundingLabel.SUPPORTED if prob >= 0.5 else GroundingLabel.NOT_SUPPORTED)
        if isinstance(dec, str):
            dec = GroundingLabel(dec)

        pred = GroundingPrediction(
            example_id=ex.id,
            decision=dec,
            probability=prob,
            evidence_spans=res.get("evidence_spans", []),
            reason=res.get("reason", "unspecified"),
            latency_us=latency_us
        )
        predictions.append(pred)

    elapsed_s = time.perf_counter() - t_start
    print(f"Completed evaluation in {elapsed_s:.2f}s ({len(examples)/max(elapsed_s, 1e-4):.1f} queries/sec).")

    # Compute comprehensive metrics
    metrics = calculate_grounding_metrics(examples, predictions)
    metrics["model_name"] = model_name
    metrics["dataset_path"] = dataset_path
    metrics["total_evaluation_time_s"] = round(elapsed_s, 2)

    # Run diagnostic error analysis
    if output_error_path:
        run_error_analysis(examples, predictions, output_path=output_error_path)

    # Save results
    if output_results_path:
        with open(output_results_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation results to {output_results_path}")

    return {
        "metrics": metrics,
        "predictions": predictions
    }
