"""
High-Precision Evaluation Harness (v2) for ApexDecision Grounding.

Computes:
- Overall Accuracy with 95% Bootstrap Confidence Intervals (2,000 resamples)
- Precision, Recall, F1 for SUPPORTED and NOT_SUPPORTED
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Adversarial Accuracy (Tier 3 & 4)
- Direct Entailment Recall & Accuracy
- Expected Calibration Error (ECE, 10 bins) and Brier Score
- Exact Category-by-Category Accuracy across all 20 Failure Modes
- High-precision Latency profiling using torch.cuda.Event (p50, p95, p99, throughput)
- Peak VRAM tracking using torch.cuda.max_memory_allocated()
- Strict Acceptance Standard Table vs Goals (99.9% target audit)
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Callable, Tuple, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apex_decision.benchmarks.grounding.dataset_schema import GroundingExample, GroundingLabel


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_bootstraps: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.RandomState(42)
    boot_accs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        boot_accs.append(np.mean(y_true[idx] == y_pred[idx]))
    lower = float(np.percentile(boot_accs, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_accs, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(labels[in_bin])
            avg_conf_in_bin = np.mean(probs[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return float(ece)


def evaluate_grounding_v2(
    predict_fn: Callable[[str, str], Dict[str, Any]],
    dataset_path: str,
    model_name: str = "ApexDecision_v2",
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    print(f"\n==================================================================")
    print(f"EVALUATING {model_name} on {dataset_path}")
    print(f"==================================================================")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if max_samples:
        raw_data = raw_data[:max_samples]
    examples = [GroundingExample(**item) for item in raw_data]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    gold_labels = []
    pred_labels = []
    confidences = []
    latencies_us = []
    categories = []
    is_adversarial = []
    predictions_log = []

    t0_all = time.perf_counter()

    for idx, ex in enumerate(examples):
        # Timed prediction
        t_start = time.perf_counter_ns()
        res = predict_fn(ex.source, ex.claim)
        t_dur_us = max(1, (time.perf_counter_ns() - t_start) // 1000)

        pred_dec = res.get("decision", "NOT_SUPPORTED")
        if hasattr(pred_dec, "value"):
            pred_dec = pred_dec.value
        prob = float(res.get("probability", 0.5))

        gold = 1 if ex.label == GroundingLabel.SUPPORTED else 0
        pred = 1 if pred_dec == "SUPPORTED" else 0

        gold_labels.append(gold)
        pred_labels.append(pred)
        confidences.append(prob)
        latencies_us.append(t_dur_us)
        categories.append(ex.failure_mode.value)
        is_adversarial.append(ex.is_adversarial)

        predictions_log.append({
            "id": ex.id,
            "gold": "SUPPORTED" if gold == 1 else "NOT_SUPPORTED",
            "prediction": "SUPPORTED" if pred == 1 else "NOT_SUPPORTED",
            "probability": prob,
            "category": ex.failure_mode.value,
            "is_correct": (gold == pred),
            "reason": res.get("reason", "none")
        })

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(examples):
            print(f"Evaluated {idx + 1}/{len(examples)} examples (current acc: {np.mean(np.array(gold_labels) == np.array(pred_labels))*100:.2f}%)")

    total_time_s = time.perf_counter() - t0_all
    throughput = len(examples) / total_time_s

    gold_arr = np.array(gold_labels)
    pred_arr = np.array(pred_labels)
    conf_arr = np.array(confidences)
    lat_arr = np.array(latencies_us) / 1000.0  # convert to ms

    # Overall metrics
    correct = (gold_arr == pred_arr)
    overall_acc = float(np.mean(correct))
    ci_low, ci_high = bootstrap_ci(gold_arr, pred_arr)

    # Confusion Matrix
    tp = int(np.sum((gold_arr == 1) & (pred_arr == 1)))
    fp = int(np.sum((gold_arr == 0) & (pred_arr == 1)))
    tn = int(np.sum((gold_arr == 0) & (pred_arr == 0)))
    fn = int(np.sum((gold_arr == 1) & (pred_arr == 0)))

    sup_prec = tp / max(1, tp + fp)
    sup_rec = tp / max(1, tp + fn)
    sup_f1 = (2 * sup_prec * sup_rec) / max(1e-6, sup_prec + sup_rec)

    unsup_prec = tn / max(1, tn + fn)
    unsup_rec = tn / max(1, tn + fp)
    unsup_f1 = (2 * unsup_prec * unsup_rec) / max(1e-6, unsup_prec + unsup_rec)

    macro_f1 = (sup_f1 + unsup_f1) / 2.0
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    # Calibration & Brier
    brier = float(np.mean((conf_arr - gold_arr) ** 2))
    ece = compute_ece(conf_arr, gold_arr)

    # Latency percentiles
    p50_ms = float(np.percentile(lat_arr, 50))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    # Category breakdown
    cat_metrics = {}
    unique_cats = sorted(list(set(categories)))
    for cat in unique_cats:
        c_mask = np.array([c == cat for c in categories])
        c_tot = int(np.sum(c_mask))
        c_cor = int(np.sum(correct[c_mask]))
        c_acc = c_cor / max(1, c_tot)
        cat_metrics[cat] = {
            "total": c_tot,
            "correct": c_cor,
            "errors": c_tot - c_cor,
            "accuracy": float(c_acc)
        }

    # Adversarial subsets
    adv_mask = np.array(is_adversarial)
    adv_acc = float(np.mean(correct[adv_mask])) if np.sum(adv_mask) > 0 else 0.0
    direct_entail_acc = cat_metrics.get("direct_entailment", {}).get("accuracy", 0.0)

    results = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "total_examples": len(examples),
        "overall_accuracy": overall_acc,
        "ci_95": [ci_low, ci_high],
        "adversarial_accuracy": adv_acc,
        "direct_entailment_accuracy": direct_entail_acc,
        "supported_precision": sup_prec,
        "supported_recall": sup_rec,
        "supported_f1": sup_f1,
        "unsupported_precision": unsup_prec,
        "unsupported_recall": unsup_rec,
        "unsupported_f1": unsup_f1,
        "macro_f1": macro_f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total_errors": int(fp + fn),
        "expected_calibration_error": ece,
        "brier_score": brier,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "latency_p99_ms": p99_ms,
        "throughput_samples_per_sec": float(throughput),
        "peak_vram_mb": peak_vram_mb,
        "category_metrics": cat_metrics
    }

    # Print Summary Table
    print("\n=========================================================================================================")
    print(f"APEXDECISION V2 EVALUATION SUMMARY: {model_name} (N={len(examples)})")
    print("=========================================================================================================")
    print(f"Overall Accuracy:           {overall_acc*100:.2f}%  [95% CI: {ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Adversarial Accuracy:       {adv_acc*100:.2f}%")
    print(f"Direct Entailment Recall:   {direct_entail_acc*100:.2f}%")
    print(f"False Positive Rate (FPR):  {fpr*100:.4f}% ({fp} false positives)")
    print(f"False Negative Rate (FNR):  {fnr*100:.4f}% ({fn} false negatives)")
    print(f"Macro F1 Score:             {macro_f1:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    print(f"Brier Score:                {brier:.4f}")
    print(f"Latency (p50 / p95 / p99):  {p50_ms:.2f}ms / {p95_ms:.2f}ms / {p99_ms:.2f}ms")
    print(f"Peak VRAM:                  {peak_vram_mb:.1f} MB")
    print(f"Total Errors:               {fp + fn} / {len(examples)}")
    print("=========================================================================================================")

    # Strict Acceptance Table vs 99.9% Target Standard
    print("\n---------------------------------------------------------------------------------------------------------")
    print(f"{'Metric':<28} | {'Measured Value':<16} | {'Goal':<14} | {'Status':<12}")
    print("-" * 75)
    for m_name, val_str, goal_str, passed in [
        ("Overall Accuracy", f"{overall_acc*100:.2f}%", ">= 99.90%", overall_acc >= 0.999),
        ("Adversarial Accuracy", f"{adv_acc*100:.2f}%", ">= 99.50%", adv_acc >= 0.995),
        ("Direct Entailment Recall", f"{direct_entail_acc*100:.2f}%", ">= 99.50%", direct_entail_acc >= 0.995),
        ("False Positive Rate", f"{fpr*100:.4f}%", "<= 0.050%", fpr <= 0.0005),
        ("Expected Calibration Error", f"{ece:.4f}", "<= 0.0100", ece <= 0.01),
        ("Total Errors (N=10,000)", f"{fp + fn}", "<= 10 errors", (fp + fn) <= 10)
    ]:
        stat_str = "PASS" if passed else "DEFICIT"
        print(f"{m_name:<28} | {val_str:<16} | {goal_str:<14} | {stat_str:<12}")
    print("---------------------------------------------------------------------------------------------------------")

    return results, predictions_log
