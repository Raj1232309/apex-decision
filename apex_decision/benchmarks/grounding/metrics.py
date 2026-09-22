"""
Evaluation metrics, statistical tests, bootstrap confidence intervals, and calibration analysis.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, brier_score_loss

from .dataset_schema import GroundingExample, GroundingPrediction, GroundingLabel


def calculate_grounding_metrics(
    examples: List[GroundingExample],
    predictions: List[GroundingPrediction],
    bootstrap_ci: bool = True,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes exhaustive grounding verification metrics:
    - Overall accuracy, precision, recall, F1 (macro and per-class)
    - Adversarial accuracy (overall and per category)
    - Domain breakdown
    - Source and claim length breakdown
    - Difficulty tier breakdown
    - Brier score and ECE across 10 empirical bins
    - Threshold-gated accuracy at [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    - 95% Bootstrap Confidence Intervals
    """
    assert len(examples) == len(predictions), "Mismatched example and prediction counts."
    n = len(examples)
    if n == 0:
        return {}

    # Map labels to binary: 1 for SUPPORTED, 0 for NOT_SUPPORTED
    y_true = np.array([1 if ex.label == GroundingLabel.SUPPORTED else 0 for ex in examples], dtype=int)
    y_pred = np.array([1 if p.decision == GroundingLabel.SUPPORTED else 0 for p in predictions], dtype=int)
    probs = np.array([p.probability for p in predictions], dtype=float)

    # 1. Overall Accuracy and Classification Report
    correct_mask = (y_pred == y_true)
    overall_acc = float(np.mean(correct_mask))

    p_sup, r_sup, f1_sup, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    p_unsup, r_unsup, f1_unsup, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    macro_f1 = float((f1_sup + f1_unsup) / 2.0)

    # Confusion matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # 2. Calibration Metrics: Brier Score and ECE (10 bins)
    brier = float(brier_score_loss(y_true, probs))

    # Binary confidence = probability if predicted 1, (1-p) if predicted 0
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    num_bins = 10
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    reliability_diagram = []

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= low) & (confidences <= high if i == num_bins - 1 else confidences < high)
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(correct_mask[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            weight = bin_count / n
            ece += weight * abs(bin_acc - bin_conf)
            reliability_diagram.append({
                "bin_idx": i,
                "range": [round(float(low), 2), round(float(high), 2)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4)
            })
        else:
            reliability_diagram.append({
                "bin_idx": i,
                "range": [round(float(low), 2), round(float(high), 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": round(float((low + high) / 2), 2)
            })

    # Accuracy at Confidence Thresholds
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    acc_at_thresholds = {}
    for th in thresholds:
        th_mask = (confidences >= th)
        if np.sum(th_mask) > 0:
            acc_at_thresholds[str(th)] = {
                "coverage": round(float(np.mean(th_mask)), 4),
                "count": int(np.sum(th_mask)),
                "accuracy": round(float(np.mean(correct_mask[th_mask])), 4)
            }
        else:
            acc_at_thresholds[str(th)] = {"coverage": 0.0, "count": 0, "accuracy": 0.0}

    # 3. Adversarial Analysis
    adv_indices = [i for i, ex in enumerate(examples) if ex.is_adversarial]
    adv_acc = float(np.mean(correct_mask[adv_indices])) if adv_indices else 0.0

    # Accuracy by Failure Mode
    fm_accuracy = {}
    from collections import defaultdict
    fm_buckets = defaultdict(list)
    for i, ex in enumerate(examples):
        fm_buckets[ex.failure_mode.value].append(correct_mask[i])

    for fm, results in fm_buckets.items():
        fm_accuracy[fm] = {
            "count": len(results),
            "accuracy": round(float(np.mean(results)), 4)
        }

    # Accuracy by Domain
    dom_accuracy = {}
    dom_buckets = defaultdict(list)
    for i, ex in enumerate(examples):
        dom_buckets[ex.domain.value].append(correct_mask[i])
    for dom, results in dom_buckets.items():
        dom_accuracy[dom] = {
            "count": len(results),
            "accuracy": round(float(np.mean(results)), 4)
        }

    # Accuracy by Difficulty Tier
    tier_accuracy = {}
    tier_buckets = defaultdict(list)
    for i, ex in enumerate(examples):
        tier_buckets[ex.difficulty_tier.value].append(correct_mask[i])
    for tier, results in tier_buckets.items():
        tier_accuracy[f"Tier_{tier}"] = {
            "count": len(results),
            "accuracy": round(float(np.mean(results)), 4)
        }

    # Accuracy by Source and Claim Length Buckets
    def get_len_bucket(word_count: int, cuts: List[int]) -> str:
        if word_count < cuts[0]: return f"<{cuts[0]}"
        if word_count < cuts[1]: return f"{cuts[0]}-{cuts[1]}"
        return f">{cuts[1]}"

    src_buckets = defaultdict(list)
    claim_buckets = defaultdict(list)
    for i, ex in enumerate(examples):
        src_len = len(ex.source.split())
        clm_len = len(ex.claim.split())
        src_buckets[get_len_bucket(src_len, [40, 70])].append(correct_mask[i])
        claim_buckets[get_len_bucket(clm_len, [15, 25])].append(correct_mask[i])

    src_len_accuracy = {b: {"count": len(r), "accuracy": round(float(np.mean(r)), 4)} for b, r in src_buckets.items()}
    claim_len_accuracy = {b: {"count": len(r), "accuracy": round(float(np.mean(r)), 4)} for b, r in claim_buckets.items()}

    # 4. Bootstrap 95% Confidence Intervals
    ci_lower, ci_upper = None, None
    if bootstrap_ci:
        rng = np.random.default_rng(seed)
        boot_accs = []
        for _ in range(n_bootstraps):
            sample_idx = rng.choice(n, size=n, replace=True)
            boot_accs.append(np.mean(correct_mask[sample_idx]))
        ci_lower = float(np.percentile(boot_accs, 2.5))
        ci_upper = float(np.percentile(boot_accs, 97.5))

    return {
        "num_samples": n,
        "overall_accuracy": round(overall_acc, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)] if ci_lower is not None else None,
        "macro_f1": round(macro_f1, 4),
        "supported_metrics": {
            "precision": round(float(p_sup), 4),
            "recall": round(float(r_sup), 4),
            "f1": round(float(f1_sup), 4),
            "count": int(np.sum(y_true == 1))
        },
        "unsupported_metrics": {
            "precision": round(float(p_unsup), 4),
            "recall": round(float(r_unsup), 4),
            "f1": round(float(f1_unsup), 4),
            "count": int(np.sum(y_true == 0))
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "adversarial_overall_accuracy": round(adv_acc, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(float(ece), 4),
        "accuracy_at_thresholds": acc_at_thresholds,
        "reliability_diagram": reliability_diagram,
        "domain_accuracy": dom_accuracy,
        "failure_mode_accuracy": fm_accuracy,
        "difficulty_tier_accuracy": tier_accuracy,
        "source_length_accuracy": src_len_accuracy,
        "claim_length_accuracy": claim_len_accuracy,
    }


def mcnemar_test(
    preds_a: List[GroundingPrediction],
    preds_b: List[GroundingPrediction],
    examples: List[GroundingExample]
) -> Dict[str, Any]:
    """
    Performs McNemar's test for paired classification evaluation on identical samples.
    chi2 = (|b - c| - 1)^2 / (b + c)
    """
    assert len(preds_a) == len(preds_b) == len(examples), "Sample counts must match."
    y_true = [ex.label == GroundingLabel.SUPPORTED for ex in examples]
    a_correct = [p.decision == ex.label for p, ex in zip(preds_a, examples)]
    b_correct = [p.decision == ex.label for p, ex in zip(preds_b, examples)]

    # b: A correct, B wrong; c: A wrong, B correct
    b = sum(1 for ac, bc in zip(a_correct, b_correct) if ac and not bc)
    c = sum(1 for ac, bc in zip(a_correct, b_correct) if not ac and bc)
    both_correct = sum(1 for ac, bc in zip(a_correct, b_correct) if ac and bc)
    both_wrong = sum(1 for ac, bc in zip(a_correct, b_correct) if not ac and not bc)

    total_discordant = b + c
    if total_discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        # Edwards continuity correction
        statistic = ((abs(b - c) - 1.0) ** 2) / float(total_discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))

    return {
        "contingency_table": {
            "both_correct": both_correct,
            "only_system_a_correct": b,
            "only_system_b_correct": c,
            "both_wrong": both_wrong
        },
        "mcnemar_chi2": round(float(statistic), 4),
        "p_value": float(p_value),
        "statistically_significant": bool(p_value < 0.05)
    }
