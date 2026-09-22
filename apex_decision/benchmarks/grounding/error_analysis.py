"""
Structured Error Analysis and Diagnostic Inspector for ApexDecision Grounding.
"""

import json
from collections import defaultdict
from typing import List, Dict, Any

from .dataset_schema import GroundingExample, GroundingPrediction, GroundingLabel


def run_error_analysis(
    examples: List[GroundingExample],
    predictions: List[GroundingPrediction],
    output_path: str = "apex_decision/benchmarks/grounding/error_analysis.json"
) -> Dict[str, Any]:
    """
    Performs comprehensive diagnostic error analysis:
    - Identifies false positives (unsupported claims wrongly verified as supported)
    - Identifies false negatives (supported claims wrongly rejected)
    - Quantifies error rate per failure mode
    - Analyzes length, domain, and confidence attributes of errors
    - Saves structured report to error_analysis.json
    """
    assert len(examples) == len(predictions), "Mismatched examples and predictions."

    false_positives = []
    false_negatives = []
    errors_by_failure_mode = defaultdict(int)
    total_by_failure_mode = defaultdict(int)
    errors_by_domain = defaultdict(int)
    total_by_domain = defaultdict(int)

    for ex, p in zip(examples, predictions):
        total_by_failure_mode[ex.failure_mode.value] += 1
        total_by_domain[ex.domain.value] += 1

        is_correct = (ex.label == p.decision)
        if not is_correct:
            errors_by_failure_mode[ex.failure_mode.value] += 1
            errors_by_domain[ex.domain.value] += 1

            err_record = {
                "id": ex.id,
                "domain": ex.domain.value,
                "failure_mode": ex.failure_mode.value,
                "difficulty_tier": ex.difficulty_tier.value,
                "source_snippet": ex.source[:120] + ("..." if len(ex.source) > 120 else ""),
                "claim": ex.claim,
                "ground_truth": ex.label.value,
                "predicted": p.decision.value,
                "predicted_probability": round(p.probability, 4),
                "latency_us": p.latency_us
            }

            if ex.label == GroundingLabel.NOT_SUPPORTED and p.decision == GroundingLabel.SUPPORTED:
                false_positives.append(err_record)
            elif ex.label == GroundingLabel.SUPPORTED and p.decision == GroundingLabel.NOT_SUPPORTED:
                false_negatives.append(err_record)

    # Sort failure modes by error rate
    failure_mode_error_rates = []
    for fm, tot in total_by_failure_mode.items():
        errs = errors_by_failure_mode[fm]
        rate = (errs / tot) if tot > 0 else 0.0
        failure_mode_error_rates.append({
            "failure_mode": fm,
            "errors": errs,
            "total": tot,
            "error_rate": round(rate, 4)
        })
    failure_mode_error_rates.sort(key=lambda x: x["error_rate"], reverse=True)

    # Sort domains by error rate
    domain_error_rates = []
    for dom, tot in total_by_domain.items():
        errs = errors_by_domain[dom]
        rate = (errs / tot) if tot > 0 else 0.0
        domain_error_rates.append({
            "domain": dom,
            "errors": errs,
            "total": tot,
            "error_rate": round(rate, 4)
        })
    domain_error_rates.sort(key=lambda x: x["error_rate"], reverse=True)

    analysis_report = {
        "summary": {
            "total_examples": len(examples),
            "total_errors": len(false_positives) + len(false_negatives),
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives),
            "top_vulnerable_failure_mode": failure_mode_error_rates[0]["failure_mode"] if failure_mode_error_rates else "none"
        },
        "error_rates_by_failure_mode": failure_mode_error_rates,
        "error_rates_by_domain": domain_error_rates,
        "top_false_positives": sorted(false_positives, key=lambda x: x["predicted_probability"], reverse=True)[:15],
        "top_false_negatives": sorted(false_negatives, key=lambda x: x["predicted_probability"])[:15]
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_report, f, indent=2)

    return analysis_report
