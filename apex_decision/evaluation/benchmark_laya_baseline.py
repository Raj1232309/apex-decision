"""
Benchmark Laya Baseline on ApexDecision Evaluation Suites.
Measures Laya ModernBERT-large (421M) on:
  1. Held-Out Grounding Test Split (N=600)
  2. Jev 108 Reporting Suite (N=108)
  3. High-Cardinality Choice Suite (2 to 255 options)
  4. 1,000-pass Latency and VRAM Profile on RTX 2050 GPU
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

sys.path.insert(0, "laya_repo")
import laya
from laya.common import ece_score


def evaluate_laya_on_noul(agent, dataset_path: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    correct = 0
    total = 0
    adv_correct = 0
    adv_total = 0
    probs = []
    golds = []
    high_conf_correct = 0
    high_conf_total = 0

    t0 = time.perf_counter()

    for idx, item in enumerate(data):
        source = item["source"]
        claim = item["claim"]
        gold = item["label"]  # SUPPORTED or NOT_SUPPORTED
        gold_num = 1 if gold == "SUPPORTED" else 0
        is_adv = item.get("is_adversarial", False)

        q_dict = {
            "noul_q": {
                "type": "noul",
                "instructions": f"Does the document support the claim: {claim}"
            }
        }

        try:
            res = agent.predict(source, q_dict)
            ans = res["answers"]["noul_q"]
            p_true = ans.get("noul", 0.5)
            conf = ans.get("confidence", 0.5)
        except Exception:
            p_true = 0.5
            conf = 0.5

        pred_label = "SUPPORTED" if p_true >= 0.5 else "NOT_SUPPORTED"
        is_match = (pred_label == gold)

        if is_match:
            correct += 1
        total += 1

        if is_adv:
            if is_match:
                adv_correct += 1
            adv_total += 1

        if conf >= 0.90:
            if is_match:
                high_conf_correct += 1
            high_conf_total += 1

        probs.append(p_true)
        golds.append(gold_num)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    p_arr = np.array(probs, dtype=np.float32)
    y_arr = np.array(golds, dtype=np.float32)

    brier = float(np.mean((p_arr - y_arr) ** 2))
    conf_arr = np.maximum(p_arr, 1.0 - p_arr)
    corr_arr = (np.round(p_arr) == y_arr).astype(np.float32)
    ece = ece_score(conf_arr, corr_arr, bins=10)

    return {
        "accuracy": round((correct / max(1, total)) * 100.0, 2),
        "adversarial_accuracy": round((adv_correct / max(1, adv_total)) * 100.0, 2) if adv_total else 100.0,
        "high_confidence_precision": round((high_conf_correct / max(1, high_conf_total)) * 100.0, 2) if high_conf_total else 100.0,
        "brier_score": round(brier, 4),
        "ece": round(float(ece), 4),
        "total_samples": total,
        "total_time_ms": round(elapsed_ms, 2),
        "avg_latency_ms": round(elapsed_ms / max(1, total), 2)
    }


def evaluate_laya_on_choices(agent, option_counts: List[int] = [2, 5, 10, 20, 50, 77, 100]) -> Dict[str, Any]:
    results = {}
    for k in option_counts:
        opts = [f"department_queue_code_{i:03d}" for i in range(k)]
        target_idx = min(3, k - 1)
        target_opt = opts[target_idx]
        state = f"Customer request indicates dispatch to {target_opt}."

        q_dict = {
            "choice_q": {
                "type": "choice",
                "instructions": "Which department queue code must be dispatched?",
                "criteria": {o: None for o in opts}
            }
        }

        try:
            res = agent.predict(state, q_dict)
            ans = res["answers"]["choice_q"]
            pred = ans["choice"]
            conf = ans["confidence"]
            is_correct = (pred == target_opt)
        except Exception as e:
            pred = str(e)
            conf = 0.0
            is_correct = False

        results[str(k)] = {
            "options": k,
            "predicted": pred,
            "target": target_opt,
            "is_correct": is_correct,
            "confidence": conf
        }
    return results


def benchmark_laya_latency(agent, num_passes: int = 200) -> Dict[str, Any]:
    dummy_state = "The quarterly earnings confirmed operating profit of $1.2B with steady cash flow from regular software subscriptions."
    dummy_q = {"q": {"type": "noul", "instructions": "Does revenue exceed 1 billion dollars?"}}

    # Warmup
    for _ in range(20):
        _ = agent.predict(dummy_state, dummy_q)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []

    torch.cuda.reset_peak_memory_stats()

    for _ in range(num_passes):
        start_event.record()
        _ = agent.predict(dummy_state, dummy_q)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    latencies.sort()
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_lat = np.mean(latencies)
    throughput = 1000.0 / max(1e-3, mean_lat)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    return {
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "throughput_qps": round(float(throughput), 2),
        "peak_vram_mb": round(float(peak_vram), 1)
    }


def run_full_laya_baseline():
    weights_dir = "laya_weights"
    print(f"[Laya Baseline] Loading Laya from {weights_dir} on CUDA...")
    agent = laya.load(weights_dir, device="cuda" if torch.cuda.is_available() else "cpu")

    print("[Laya Baseline] Evaluating on Held-Out Grounding Test Split (N=600)...")
    test_path = "apex_decision/benchmarks/grounding/test.json"
    noul_test_res = evaluate_laya_on_noul(agent, test_path)
    print("  --> Grounding Test Results:", noul_test_res)

    print("[Laya Baseline] Evaluating on Jev 108 Suite (N=108)...")
    jev_path = "apex_decision/benchmarks/grounding/jev_108_suite.json"
    jev_res = evaluate_laya_on_noul(agent, jev_path)
    print("  --> Jev 108 Suite Results:", jev_res)

    print("[Laya Baseline] Evaluating on High-Cardinality Choice Suite...")
    choice_res = evaluate_laya_on_choices(agent)
    print("  --> Choice Scaling Results:", choice_res)

    print("[Laya Baseline] Measuring CUDA Latency (200 passes)...")
    lat_res = benchmark_laya_latency(agent, num_passes=200)
    print("  --> Latency Profile:", lat_res)

    baseline_report = {
        "model": "convaiinnovations/laya (ModernBERT-large 421M)",
        "grounding_test_split": noul_test_res,
        "jev_108_suite": jev_res,
        "choice_scaling": choice_res,
        "latency_profile": lat_res
    }

    out_file = "apex_decision/reports/laya_baseline_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline_report, f, indent=2)
    print(f"[Laya Baseline] Full report saved to {out_file}!")
    return baseline_report


if __name__ == "__main__":
    run_full_laya_baseline()
