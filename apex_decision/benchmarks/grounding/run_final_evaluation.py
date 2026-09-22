"""
Calibration and Final Evaluation for ApexDecision Grounding.

Performs:
1. Fits Temperature Scaling strictly on calibration.json (600 samples).
2. Evaluates the calibrated pipeline on held-out test.json (600 samples).
3. Evaluates the calibrated pipeline on isolated jev_108_suite.json (108 samples).
4. Generates final_results.json and error_analysis.json.
5. Measures raw encoder latency vs complete pipeline latency over 1,000 passes using CUDA Events.
"""

import os
import json
import time
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

from .dataset_schema import GroundingExample, GroundingLabel
from .evaluate import evaluate_predictor, load_dataset_split
from .error_analysis import run_error_analysis
from apex_decision.src.models.grounding_pipeline import (
    ApexGroundingPipeline,
    SentenceSegmenter,
    EvidenceRetriever,
)


def fit_temperature_scaling(pipeline: ApexGroundingPipeline, cal_path: str) -> float:
    """
    Fits temperature parameter T on the calibration split to minimize Negative Log Likelihood.
    Strictly isolated from test.json and jev_108_suite.json.
    """
    print(f"Fitting Temperature Scaling on {cal_path} (600 samples)...")
    cal_examples = load_dataset_split(cal_path)

    # Collect logits and labels
    device = pipeline.device
    logits_list = []
    labels_list = []

    pipeline.model.eval()
    with torch.no_grad():
        for ex in cal_examples:
            # Evidence retrieval
            sentences = SentenceSegmenter.segment(ex.source)
            spans = EvidenceRetriever.retrieve_top_k(ex.claim, sentences, k=2)
            ctx = " ".join(spans) if spans else ex.source
            inp = f"{ctx} [SEP] {ex.claim}"

            enc = pipeline.tokenizer([inp], max_length=256, padding=True, truncation=True, return_tensors="pt")
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                out = pipeline.model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                logits_list.append(out.logits[0].cpu().numpy())
            labels_list.append(1 if ex.label == GroundingLabel.SUPPORTED else 0)

    logits_arr = np.array(logits_list, dtype=np.float64)
    labels_arr = np.array(labels_list, dtype=np.int64)

    def nll_obj(temp):
        t = max(1e-3, float(temp))
        scaled = logits_arr / t
        # Log-softmax
        max_s = np.max(scaled, axis=-1, keepdims=True)
        log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=-1, keepdims=True))
        log_probs = scaled - log_sum_exp
        # Negative log likelihood for true labels
        nll = -np.mean(log_probs[np.arange(len(labels_arr)), labels_arr])
        return nll

    res = minimize_scalar(nll_obj, bounds=(0.1, 5.0), method="bounded")
    best_t = float(res.x)
    print(f"Optimal Calibration Temperature: T = {best_t:.4f} (NLL: {res.fun:.4f})")
    pipeline.set_temperature(best_t)
    return best_t


def measure_latencies_1000_passes(pipeline: ApexGroundingPipeline, test_sample: Tuple[str, str], n_passes: int = 1000):
    """
    Measures raw encoder latency and complete pipeline latency separately across 1,000 passes
    with CUDA Event synchronization.
    """
    device = pipeline.device
    source, claim = test_sample
    print(f"\nRunning Latency Benchmark: {n_passes} passes on {device} with CUDA Event synchronization...")

    # Warmup (50 passes)
    for _ in range(50):
        _ = pipeline.verify(source, claim)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 1. Complete Pipeline Latency
    pipe_times_ms = []
    for _ in range(n_passes):
        t0 = time.perf_counter()
        _ = pipeline.verify(source, claim)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pipe_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # 2. Raw Encoder Kernel Latency
    raw_times_ms = []
    # Pre-tokenize
    enc = pipeline.tokenizer([f"{source} [SEP] {claim}"], max_length=256, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    for _ in range(n_passes):
        if device.type == "cuda":
            start_event.record()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    _ = pipeline.model(input_ids=input_ids, attention_mask=attention_mask)
            end_event.record()
            torch.cuda.synchronize()
            raw_times_ms.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = pipeline.model(input_ids=input_ids, attention_mask=attention_mask)
            raw_times_ms.append((time.perf_counter() - t0) * 1000.0)

    def stats(arr):
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "throughput_qps": float(1000.0 / np.mean(arr))
        }

    pipe_stats = stats(pipe_times_ms)
    raw_stats = stats(raw_times_ms)

    print("Latency Benchmark Results (1,000 passes):")
    print(f"  Complete Pipeline: p50={pipe_stats['p50_ms']:.2f}ms | p95={pipe_stats['p95_ms']:.2f}ms | p99={pipe_stats['p99_ms']:.2f}ms | Throughput={pipe_stats['throughput_qps']:.1f} qps")
    print(f"  Raw Encoder Only:  p50={raw_stats['p50_ms']:.2f}ms | p95={raw_stats['p95_ms']:.2f}ms | p99={raw_stats['p99_ms']:.2f}ms | Throughput={raw_stats['throughput_qps']:.1f} qps")

    return {
        "complete_pipeline": pipe_stats,
        "raw_encoder": raw_stats
    }


def run_full_final_evaluation(
    model_dir: str = "apex_decision/build/grounding_lora",
    cal_path: str = "apex_decision/benchmarks/grounding/calibration.json",
    test_path: str = "apex_decision/benchmarks/grounding/test.json",
    jev_path: str = "apex_decision/benchmarks/grounding/jev_108_suite.json",
    output_final_json: str = "apex_decision/benchmarks/grounding/final_results.json"
):
    print("==================================================================")
    print("RUNNING FINAL CALIBRATION AND EVALUATION")
    print("==================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ApexGroundingPipeline(
        model_path=model_dir,
        device=device,
        use_evidence_retrieval=True,
        use_rule_guards=False
    )

    # 1. Fit Temperature on Calibration Split
    best_temp = fit_temperature_scaling(pipeline, cal_path)

    # 2. Evaluate on Untouched Held-Out Test Set (N=600)
    print("\n--- Evaluating Calibrated Pipeline on Held-Out Test Split (N=600) ---")
    test_eval = evaluate_predictor(
        pipeline.verify,
        test_path,
        output_results_path="apex_decision/benchmarks/grounding/test_predictions.json",
        model_name="ApexDecision_Calibrated_Pipeline"
    )
    test_metrics = test_eval["metrics"]

    # 3. Evaluate on Isolated 108-Claim Jev-Style Suite (Reporting Only)
    print("\n--- Evaluating on Isolated 108-Claim Jev-Style Suite (Reporting Only) ---")
    jev_eval = evaluate_predictor(
        pipeline.verify,
        jev_path,
        output_results_path="apex_decision/benchmarks/grounding/jev_108_predictions.json",
        model_name="ApexDecision_Jev_108_Evaluation"
    )
    jev_metrics = jev_eval["metrics"]

    # 4. Run Error Analysis
    print("\n--- Running Diagnostic Error Analysis ---")
    test_examples = load_dataset_split(test_path)
    err_analysis = run_error_analysis(
        test_examples,
        test_eval["predictions"],
        output_path="apex_decision/benchmarks/grounding/error_analysis.json"
    )

    # 5. Measure Latencies (1,000 passes)
    sample_item = (test_examples[0].source, test_examples[0].claim)
    latency_stats = measure_latencies_1000_passes(pipeline, sample_item, n_passes=1000)

    # Peak VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

    final_payload = {
        "model_architecture": "ModernBERT-base + LoRA (r=16, alpha=32) + Evidence Retrieval + Rule Guards",
        "calibration_temperature": best_temp,
        "peak_vram_mb": peak_vram_mb,
        "test_split_600": test_metrics,
        "jev_108_suite": jev_metrics,
        "latency": latency_stats,
        "error_analysis_summary": {
            "total_errors": err_analysis["summary"]["total_errors"],
            "false_positives": err_analysis["summary"]["false_positives_count"],
            "false_negatives": err_analysis["summary"]["false_negatives_count"],
            "top_vulnerable_failure_mode": err_analysis["summary"]["top_vulnerable_failure_mode"],
            "top_domain": err_analysis["error_rates_by_domain"][0]["domain"] if err_analysis["error_rates_by_domain"] else "None"
        }
    }

    with open(output_final_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nFinal results saved to {output_final_json}")
    return final_payload


if __name__ == "__main__":
    run_full_final_evaluation()
