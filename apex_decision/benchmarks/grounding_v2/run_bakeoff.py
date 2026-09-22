"""
Controlled Backbone Bake-Off for ApexDecision Grounding.

Compares under identical conditions:
Candidate A: answerdotai/ModernBERT-base + LoRA
Candidate B: microsoft/deberta-v3-base + LoRA
Candidate C: cross-encoder/nli-deberta-v3-base (adapted to grounding)

Evaluates on identical validation.json and test.json splits.
Measures:
- Overall Accuracy
- Adversarial Accuracy (Tier 3 & 4)
- Direct Entailment Accuracy & Recall
- Contradiction Accuracy
- False Positives (FPR) & False Negatives (FNR)
- Macro F1
- Brier Score & ECE
- Inference Latency (p50, p95, p99) via CUDA Events
- Peak VRAM (strictly < 3,500 MB)
- Trainable & Inference Parameter Counts
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apex_decision.benchmarks.grounding.dataset_schema import GroundingExample, GroundingLabel
from apex_decision.benchmarks.grounding_v2.evaluate_v2 import bootstrap_ci, compute_ece


def profile_model_inference(
    model,
    tokenizer,
    dataset: List[GroundingExample],
    device: str,
    model_name: str,
    is_3class_nli: bool = False
) -> Dict[str, Any]:
    print(f"\nEvaluating {model_name} (N={len(dataset)})...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.eval()
    gold_labels = []
    pred_labels = []
    probabilities = []
    categories = []
    is_adv = []
    latencies_us = []

    # CUDA Event Latency measurement
    start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    # Warmup
    dummy = tokenizer(["Test source text [SEP] Test claim"], max_length=256, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(**dummy)

    t0_start = time.perf_counter()

    for idx, ex in enumerate(dataset):
        text = f"{ex.source} [SEP] {ex.claim}"
        enc = tokenizer([text], max_length=256, padding=True, truncation=True, return_tensors="pt").to(device)

        if device == "cuda":
            start_event.record()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
            latencies_us.append(int(latency_ms * 1000))
        else:
            t1 = time.perf_counter_ns()
            with torch.no_grad():
                outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            latencies_us.append(max(1, (time.perf_counter_ns() - t1) // 1000))

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        if is_3class_nli:
            # 3-class NLI mapping:
            # idx 0: contradiction, idx 1: entailment, idx 2: neutral
            # SUPPORTED = entailment; NOT_SUPPORTED = contradiction or neutral
            prob_sup = float(probs[1])
            pred = 1 if prob_sup >= 0.50 else 0
        else:
            # 2-class classifier: idx 0 = NOT_SUPPORTED, idx 1 = SUPPORTED
            prob_sup = float(probs[1])
            pred = 1 if prob_sup >= 0.50 else 0

        gold = 1 if ex.label == GroundingLabel.SUPPORTED else 0
        gold_labels.append(gold)
        pred_labels.append(pred)
        probabilities.append(prob_sup)
        categories.append(ex.failure_mode.value)
        is_adv.append(ex.is_adversarial)

    total_time_s = time.perf_counter() - t0_start
    throughput = len(dataset) / total_time_s

    gold_arr = np.array(gold_labels)
    pred_arr = np.array(pred_labels)
    conf_arr = np.array(probabilities)
    lat_arr = np.array(latencies_us) / 1000.0  # ms

    correct = (gold_arr == pred_arr)
    overall_acc = float(np.mean(correct))
    ci_low, ci_high = bootstrap_ci(gold_arr, pred_arr)

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

    brier = float(np.mean((conf_arr - gold_arr) ** 2))
    ece = compute_ece(conf_arr, gold_arr)

    p50_ms = float(np.percentile(lat_arr, 50))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    # Category accuracies
    cat_metrics = {}
    unique_cats = sorted(list(set(categories)))
    for cat in unique_cats:
        c_mask = np.array([c == cat for c in categories])
        c_tot = int(np.sum(c_mask))
        c_cor = int(np.sum(correct[c_mask]))
        cat_metrics[cat] = {
            "total": c_tot,
            "correct": c_cor,
            "accuracy": float(c_cor / max(1, c_tot))
        }

    adv_mask = np.array(is_adv)
    adv_acc = float(np.mean(correct[adv_mask])) if np.sum(adv_mask) > 0 else 0.0
    direct_entail_acc = cat_metrics.get("direct_entailment", {}).get("accuracy", 0.0)
    contra_acc = cat_metrics.get("direct_contradiction", {}).get("accuracy", 0.0)

    # Param counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "model_name": model_name,
        "total_examples": len(dataset),
        "overall_accuracy": overall_acc,
        "ci_95": [ci_low, ci_high],
        "adversarial_accuracy": adv_acc,
        "direct_entailment_accuracy": direct_entail_acc,
        "direct_contradiction_accuracy": contra_acc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "false_positives": fp,
        "false_negatives": fn,
        "supported_precision": sup_prec,
        "supported_recall": sup_rec,
        "supported_f1": sup_f1,
        "macro_f1": macro_f1,
        "expected_calibration_error": ece,
        "brier_score": brier,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "latency_p99_ms": p99_ms,
        "throughput": float(throughput),
        "peak_vram_mb": peak_vram_mb,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "category_metrics": cat_metrics
    }
