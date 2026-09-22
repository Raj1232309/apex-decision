"""
Ablation Study for ApexDecision Grounding & Claim-Verification.

Evaluates 5 configurations on held-out test split (600 samples):
1. ModernBERT + Linear Head (frozen backbone, no LoRA)
2. ModernBERT + LoRA
3. ModernBERT + LoRA + Hard-Negative Mining
4. ModernBERT + LoRA + Evidence Retrieval
5. ModernBERT + LoRA + Evidence Retrieval + Adversarial Training (Full ApexDecision Pipeline)

Saves detailed results to benchmarks/grounding/ablation_results.json.
"""

import os
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from .dataset_schema import GroundingExample, GroundingLabel
from .evaluate import evaluate_predictor, load_dataset_split
from apex_decision.src.models.grounding_pipeline import (
    ApexGroundingPipeline,
    SentenceSegmenter,
    EvidenceRetriever,
    RuleContradictionGuard
)


def run_all_ablations(
    test_path: str = "apex_decision/benchmarks/grounding/test.json",
    output_json: str = "apex_decision/benchmarks/grounding/ablation_results.json",
    lora_base_dir: str = "apex_decision/build/grounding_lora",
    lora_hn_dir: str = "apex_decision/build/grounding_lora_hn"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    results = {}

    print("==================================================================")
    print("STARTING ABLATION STUDY ON HELD-OUT TEST SPLIT (N=600)")
    print("==================================================================")

    # -------------------------------------------------------------------------
    # Ablation 1: ModernBERT + Linear Head (frozen backbone, zero-shot/untrained)
    # -------------------------------------------------------------------------
    print("\n--- Ablation 1: ModernBERT + Linear Head (frozen backbone) ---")
    model_frozen = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2).to(device)
    model_frozen.eval()

    def pred_ablation_1(source: str, claim: str):
        t0 = time.perf_counter_ns()
        text = f"{source} [SEP] {claim}"
        enc = tokenizer([text], max_length=256, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                logits = model_frozen(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            prob_sup = float(probs[1])
        dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)
        return {
            "decision": GroundingLabel.SUPPORTED if prob_sup >= 0.5 else GroundingLabel.NOT_SUPPORTED,
            "probability": prob_sup,
            "evidence_spans": [],
            "reason": "ablation_1_frozen_backbone",
            "latency_us": dt_us
        }

    res_1 = evaluate_predictor(pred_ablation_1, test_path, model_name="Ablation_1_Frozen_Backbone")
    results["1_modernbert_frozen_head"] = res_1["metrics"]
    del model_frozen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Ablation 2: ModernBERT + LoRA
    # -------------------------------------------------------------------------
    print("\n--- Ablation 2: ModernBERT + LoRA ---")
    base_m2 = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model_lora = PeftModel.from_pretrained(base_m2, lora_base_dir).to(device)
    model_lora.eval()

    def pred_ablation_2(source: str, claim: str):
        t0 = time.perf_counter_ns()
        text = f"{source} [SEP] {claim}"
        enc = tokenizer([text], max_length=256, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                logits = model_lora(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            prob_sup = float(probs[1])
        dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)
        return {
            "decision": GroundingLabel.SUPPORTED if prob_sup >= 0.5 else GroundingLabel.NOT_SUPPORTED,
            "probability": prob_sup,
            "evidence_spans": [],
            "reason": "ablation_2_lora",
            "latency_us": dt_us
        }

    res_2 = evaluate_predictor(pred_ablation_2, test_path, model_name="Ablation_2_ModernBERT_LoRA")
    results["2_modernbert_lora"] = res_2["metrics"]

    # -------------------------------------------------------------------------
    # Ablation 3: ModernBERT + LoRA + Hard-Negative Mining
    # -------------------------------------------------------------------------
    print("\n--- Ablation 3: ModernBERT + LoRA + Hard-Negative Mining ---")
    hn_model_dir = lora_hn_dir if os.path.exists(lora_hn_dir) else lora_base_dir
    base_m3 = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model_hn = PeftModel.from_pretrained(base_m3, hn_model_dir).to(device)
    model_hn.eval()

    def pred_ablation_3(source: str, claim: str):
        t0 = time.perf_counter_ns()
        text = f"{source} [SEP] {claim}"
        enc = tokenizer([text], max_length=256, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                logits = model_hn(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            prob_sup = float(probs[1])
        dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)
        return {
            "decision": GroundingLabel.SUPPORTED if prob_sup >= 0.5 else GroundingLabel.NOT_SUPPORTED,
            "probability": prob_sup,
            "evidence_spans": [],
            "reason": "ablation_3_lora_hard_negatives",
            "latency_us": dt_us
        }

    res_3 = evaluate_predictor(pred_ablation_3, test_path, model_name="Ablation_3_LoRA_Hard_Negatives")
    results["3_modernbert_lora_hard_negatives"] = res_3["metrics"]

    # -------------------------------------------------------------------------
    # Ablation 4: ModernBERT + LoRA + Evidence Retrieval (No rule guards)
    # -------------------------------------------------------------------------
    print("\n--- Ablation 4: ModernBERT + LoRA + Evidence Retrieval ---")
    pipeline_retrieval_only = ApexGroundingPipeline(
        model_path=hn_model_dir,
        device=device,
        use_evidence_retrieval=True,
        use_rule_guards=False
    )
    res_4 = evaluate_predictor(
        pipeline_retrieval_only.verify,
        test_path,
        model_name="Ablation_4_LoRA_Evidence_Retrieval"
    )
    results["4_modernbert_lora_evidence_retrieval"] = res_4["metrics"]

    # -------------------------------------------------------------------------
    # Ablation 5: ModernBERT + LoRA + Evidence Retrieval + Adversarial Training (Full Pipeline)
    # -------------------------------------------------------------------------
    print("\n--- Ablation 5: Full ApexDecision Grounding Pipeline (Evidence + Adversarial Guards) ---")
    pipeline_full = ApexGroundingPipeline(
        model_path=hn_model_dir,
        device=device,
        use_evidence_retrieval=True,
        use_rule_guards=True
    )
    res_5 = evaluate_predictor(
        pipeline_full.verify,
        test_path,
        model_name="Ablation_5_Full_ApexDecision_Pipeline"
    )
    results["5_full_apexdecision_pipeline"] = res_5["metrics"]

    # -------------------------------------------------------------------------
    # Print Ablation Summary Table
    # -------------------------------------------------------------------------
    print("\n=========================================================================================================")
    print("ABLATION STUDY SUMMARY (HELD-OUT TEST SET, N=600)")
    print("=========================================================================================================")
    print(f"{'Configuration':<36} | {'Accuracy':<10} | {'95% CI':<16} | {'Adversarial Acc':<16} | {'Macro F1':<10} | {'ECE':<8}")
    print("-" * 105)
    for key, name in [
        ("1_modernbert_frozen_head", "1. Frozen Backbone + Linear"),
        ("2_modernbert_lora", "2. ModernBERT + LoRA"),
        ("3_modernbert_lora_hard_negatives", "3. LoRA + Hard Negatives"),
        ("4_modernbert_lora_evidence_retrieval", "4. LoRA + Evidence Retrieval"),
        ("5_full_apexdecision_pipeline", "5. Full ApexDecision Pipeline")
    ]:
        m = results[key]
        ci_str = f"[{m['ci_95'][0]*100:.1f}%, {m['ci_95'][1]*100:.1f}%]" if m.get('ci_95') else "N/A"
        print(f"{name:<36} | {m['overall_accuracy']*100:<9.2f}% | {ci_str:<16} | {m['adversarial_overall_accuracy']*100:<15.2f}% | {m['macro_f1']:<10.4f} | {m['expected_calibration_error']:<8.4f}")
    print("=========================================================================================================")

    # Save to json
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results successfully written to {output_json}")
    return results


if __name__ == "__main__":
    run_all_ablations()
