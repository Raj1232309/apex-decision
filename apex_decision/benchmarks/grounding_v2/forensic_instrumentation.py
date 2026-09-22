"""
Forensic Architecture Audit Instrumentation for ApexDecision Grounding.

Inspects:
1. Model checkpoint hashes (grounding_lora vs grounding_lora_hn).
2. Training dataset hashes (train.json vs train_augmented.json).
3. Sentence segmentation and evidence retrieval on test.json (tokenized input comparison).
4. Full inference tracing: raw input, candidate sentences, retrieved evidence, token IDs, logits, probs, predictions.
5. Prediction hash and probability delta comparison across ablations.
"""

import os
import sys
import json
import hashlib
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Adjust sys.path to allow imports from repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apex_decision.benchmarks.grounding.dataset_schema import GroundingExample, GroundingLabel
from apex_decision.src.models.grounding_pipeline import (
    SentenceSegmenter,
    EvidenceRetriever,
    RuleContradictionGuard,
    ApexGroundingPipeline,
)


def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_directory_hash(dirpath: str) -> Dict[str, str]:
    if not os.path.exists(dirpath):
        return {"status": "NOT_FOUND"}
    file_hashes = {}
    for root, _, files in os.walk(dirpath):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, dirpath)
            file_hashes[rel] = compute_file_sha256(p)
    return file_hashes


def run_forensic_audit():
    print("==================================================================")
    print("APEXDECISION V2: FORENSIC ARCHITECTURE AUDIT INSTRUMENTATION")
    print("==================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)

    test_path = "apex_decision/benchmarks/grounding/test.json"
    train_path = "apex_decision/benchmarks/grounding/train.json"
    train_aug_path = "apex_decision/benchmarks/grounding/train_augmented.json"
    lora_dir = "apex_decision/build/grounding_lora"
    lora_hn_dir = "apex_decision/build/grounding_lora_hn"

    # Step 1: Checkpoint & Data File Integrity Hashes
    print("\n--- Step 1: Checkpoint and Data Hashes ---")
    train_hash = compute_file_sha256(train_path)
    train_aug_hash = compute_file_sha256(train_aug_path)
    print(f"train.json SHA-256:           {train_hash}")
    print(f"train_augmented.json SHA-256: {train_aug_hash}")
    train_identical = (train_hash == train_aug_hash)
    print(f"Are train.json and train_augmented.json identical? {train_identical}")

    lora_files = compute_directory_hash(lora_dir)
    lora_hn_files = compute_directory_hash(lora_hn_dir)
    print(f"LoRA files in {lora_dir}: {list(lora_files.keys())}")
    print(f"LoRA HN files in {lora_hn_dir}: {list(lora_hn_files.keys())}")

    weights_identical = True
    for f, h in lora_files.items():
        hn_h = lora_hn_files.get(f)
        if h != hn_h:
            weights_identical = False
            print(f"Difference in {f}: {h} != {hn_h}")
    print(f"Are grounding_lora and grounding_lora_hn weights identical? {weights_identical}")

    # Step 2: Load Test Set & Sentence Analysis
    print("\n--- Step 2: Source Text Sentence Length Analysis (N=600) ---")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_examples = [GroundingExample(**item) for item in test_data]

    sentence_counts = []
    sources_le_2_sentences = 0
    sources_gt_2_sentences = 0
    evidence_retrieval_diff_count = 0
    tokenized_input_diff_count = 0

    retrieval_samples = []

    for idx, ex in enumerate(test_examples):
        sents = SentenceSegmenter.segment(ex.source)
        sentence_counts.append(len(sents))
        if len(sents) <= 2:
            sources_le_2_sentences += 1
        else:
            sources_gt_2_sentences += 1

        top_k = EvidenceRetriever.retrieve_top_k(ex.claim, sents, k=2)
        retrieved_context = " ".join(top_k) if top_k else ex.source

        # Check if retrieved text differs from raw source text
        is_retrieved_diff = (retrieved_context.strip() != ex.source.strip())
        if is_retrieved_diff:
            evidence_retrieval_diff_count += 1

        # Check tokenized inputs
        raw_text = f"{ex.source} [SEP] {ex.claim}"
        ret_text = f"{retrieved_context} [SEP] {ex.claim}"

        raw_ids = tokenizer.encode(raw_text, max_length=256, truncation=True)
        ret_ids = tokenizer.encode(ret_text, max_length=256, truncation=True)

        if raw_ids != ret_ids:
            tokenized_input_diff_count += 1

        if idx < 5 or (is_retrieved_diff and len(retrieval_samples) < 8):
            retrieval_samples.append({
                "example_id": ex.id,
                "domain": ex.domain.value,
                "category": ex.failure_mode.value,
                "source_num_sents": len(sents),
                "raw_source": ex.source,
                "candidate_sentences": sents,
                "retrieved_evidence": top_k,
                "retrieved_context": retrieved_context,
                "claim": ex.claim,
                "is_context_different": is_retrieved_diff,
                "raw_token_count": len(raw_ids),
                "ret_token_count": len(ret_ids),
                "raw_token_ids_prefix": raw_ids[:10],
                "ret_token_ids_prefix": ret_ids[:10]
            })

    print(f"Sources with <= 2 sentences: {sources_le_2_sentences} / {len(test_examples)} ({sources_le_2_sentences/len(test_examples)*100:.2f}%)")
    print(f"Sources with > 2 sentences:  {sources_gt_2_sentences} / {len(test_examples)} ({sources_gt_2_sentences/len(test_examples)*100:.2f}%)")
    print(f"Cases where retrieved context string != raw source: {evidence_retrieval_diff_count} / {len(test_examples)} ({evidence_retrieval_diff_count/len(test_examples)*100:.2f}%)")
    print(f"Cases where tokenized input tensor differed:         {tokenized_input_diff_count} / {len(test_examples)} ({tokenized_input_diff_count/len(test_examples)*100:.2f}%)")

    # Step 3: Full Inference Tracing on Model Checkpoints
    print("\n--- Step 3: Inference Tracing across Ablations ---")
    base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model_lora = PeftModel.from_pretrained(base_model, lora_dir).to(device)
    model_lora.eval()

    # If lora_hn_dir exists and differs, load it; otherwise we know it's identical
    base_model_hn = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model_hn = PeftModel.from_pretrained(base_model_hn, lora_hn_dir).to(device)
    model_hn.eval()

    # Trace predictions
    traces_abl2 = [] # LoRA
    traces_abl3 = [] # LoRA + Hard Negatives
    traces_abl4 = [] # LoRA + Evidence Retrieval

    batch_size = 16
    for i in range(0, len(test_examples), batch_size):
        batch = test_examples[i:i + batch_size]

        # Ablation 2 inputs: raw source [SEP] claim
        texts_raw = [f"{ex.source} [SEP] {ex.claim}" for ex in batch]
        enc_raw = tokenizer(texts_raw, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

        # Ablation 4 inputs: retrieved evidence [SEP] claim
        contexts_ret = []
        for ex in batch:
            s = SentenceSegmenter.segment(ex.source)
            top_k = EvidenceRetriever.retrieve_top_k(ex.claim, s, k=2)
            contexts_ret.append(" ".join(top_k) if top_k else ex.source)
        texts_ret = [f"{c} [SEP] {ex.claim}" for c, ex in zip(contexts_ret, batch)]
        enc_ret = tokenizer(texts_ret, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                # Ablation 2
                out2 = model_lora(input_ids=enc_raw["input_ids"], attention_mask=enc_raw["attention_mask"])
                logits2 = out2.logits.cpu().numpy()
                probs2 = torch.softmax(torch.tensor(logits2), dim=-1).numpy()

                # Ablation 3
                out3 = model_hn(input_ids=enc_raw["input_ids"], attention_mask=enc_raw["attention_mask"])
                logits3 = out3.logits.cpu().numpy()
                probs3 = torch.softmax(torch.tensor(logits3), dim=-1).numpy()

                # Ablation 4
                out4 = model_lora(input_ids=enc_ret["input_ids"], attention_mask=enc_ret["attention_mask"])
                logits4 = out4.logits.cpu().numpy()
                probs4 = torch.softmax(torch.tensor(logits4), dim=-1).numpy()

        for j in range(len(batch)):
            traces_abl2.append({
                "logits": logits2[j].tolist(),
                "prob_supported": float(probs2[j, 1]),
                "pred": "SUPPORTED" if probs2[j, 1] >= 0.5 else "NOT_SUPPORTED"
            })
            traces_abl3.append({
                "logits": logits3[j].tolist(),
                "prob_supported": float(probs3[j, 1]),
                "pred": "SUPPORTED" if probs3[j, 1] >= 0.5 else "NOT_SUPPORTED"
            })
            traces_abl4.append({
                "logits": logits4[j].tolist(),
                "prob_supported": float(probs4[j, 1]),
                "pred": "SUPPORTED" if probs4[j, 1] >= 0.5 else "NOT_SUPPORTED"
            })

    # Step 4: Compare Predictions and Probabilities
    print("\n--- Step 4: Prediction & Probability Comparisons ---")
    preds2 = [t["pred"] for t in traces_abl2]
    preds3 = [t["pred"] for t in traces_abl3]
    preds4 = [t["pred"] for t in traces_abl4]

    probs2 = np.array([t["prob_supported"] for t in traces_abl2])
    probs3 = np.array([t["prob_supported"] for t in traces_abl3])
    probs4 = np.array([t["prob_supported"] for t in traces_abl4])

    diff_preds_2_vs_3 = sum(p2 != p3 for p2, p3 in zip(preds2, preds3))
    diff_preds_2_vs_4 = sum(p2 != p4 for p2, p4 in zip(preds2, preds4))

    mean_abs_prob_diff_2_vs_3 = float(np.mean(np.abs(probs2 - probs3)))
    mean_abs_prob_diff_2_vs_4 = float(np.mean(np.abs(probs2 - probs4)))

    max_prob_diff_2_vs_3 = float(np.max(np.abs(probs2 - probs3)))
    max_prob_diff_2_vs_4 = float(np.max(np.abs(probs2 - probs4)))

    pred_hash_2 = hashlib.sha256("".join(preds2).encode()).hexdigest()
    pred_hash_3 = hashlib.sha256("".join(preds3).encode()).hexdigest()
    pred_hash_4 = hashlib.sha256("".join(preds4).encode()).hexdigest()

    print(f"Ablation 2 Prediction Hash: {pred_hash_2}")
    print(f"Ablation 3 Prediction Hash: {pred_hash_3}")
    print(f"Ablation 4 Prediction Hash: {pred_hash_4}")
    print(f"Ablation 2 vs 3 Differing Predictions: {diff_preds_2_vs_3} / 600")
    print(f"Ablation 2 vs 3 Mean Abs Prob Diff:     {mean_abs_prob_diff_2_vs_3:.8f}")
    print(f"Ablation 2 vs 3 Max Abs Prob Diff:      {max_prob_diff_2_vs_3:.8f}")
    print(f"Ablation 2 vs 4 Differing Predictions: {diff_preds_2_vs_4} / 600")
    print(f"Ablation 2 vs 4 Mean Abs Prob Diff:     {mean_abs_prob_diff_2_vs_4:.8f}")
    print(f"Ablation 2 vs 4 Max Abs Prob Diff:      {max_prob_diff_2_vs_4:.8f}")

    # Compile Audit Results
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "integrity_checks": {
            "train_file": train_path,
            "train_augmented_file": train_aug_path,
            "train_sha256": train_hash,
            "train_augmented_sha256": train_aug_hash,
            "train_and_augmented_identical": train_identical,
            "lora_weights_hash": lora_files,
            "lora_hn_weights_hash": lora_hn_files,
            "lora_and_hn_weights_identical": weights_identical
        },
        "retrieval_instrumentation": {
            "total_test_examples": len(test_examples),
            "sources_le_2_sentences": sources_le_2_sentences,
            "sources_gt_2_sentences": sources_gt_2_sentences,
            "retrieved_context_differed_from_source_count": evidence_retrieval_diff_count,
            "tokenized_input_ids_differed_count": tokenized_input_diff_count,
            "retrieval_samples": retrieval_samples
        },
        "ablation_comparison": {
            "ablation_2_lora": {
                "checkpoint": lora_dir,
                "prediction_hash": pred_hash_2,
                "accuracy": float(np.mean([p == (ex.label.value) for p, ex in zip(preds2, test_examples)]))
            },
            "ablation_3_lora_hn": {
                "checkpoint": lora_hn_dir,
                "prediction_hash": pred_hash_3,
                "differing_predictions_vs_abl2": diff_preds_2_vs_3,
                "mean_abs_prob_diff_vs_abl2": mean_abs_prob_diff_2_vs_3,
                "max_prob_diff_vs_abl2": max_prob_diff_2_vs_3,
                "accuracy": float(np.mean([p == (ex.label.value) for p, ex in zip(preds3, test_examples)]))
            },
            "ablation_4_lora_retrieval": {
                "checkpoint": lora_dir,
                "prediction_hash": pred_hash_4,
                "differing_predictions_vs_abl2": diff_preds_2_vs_4,
                "mean_abs_prob_diff_vs_abl2": mean_abs_prob_diff_2_vs_4,
                "max_prob_diff_vs_abl2": max_prob_diff_2_vs_4,
                "accuracy": float(np.mean([p == (ex.label.value) for p, ex in zip(preds4, test_examples)]))
            }
        }
    }

    out_file = "apex_decision/benchmarks/grounding_v2/forensic_audit_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nForensic audit data successfully written to {out_file}")
    return audit_data


if __name__ == "__main__":
    run_forensic_audit()
