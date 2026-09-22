"""
Baseline Models Evaluation for ApexDecision Grounding Benchmark.

Evaluates:
1. Lexical Word-Overlap / Jaccard Baseline
2. TF-IDF + Logistic Regression Baseline
3. Legacy Pre-Trained ApexDecision Model (ModernBERT without grounding adaptation)
"""

import os
import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .dataset_schema import GroundingExample, GroundingLabel
from .evaluate import evaluate_predictor, load_dataset_split


def build_lexical_baseline(threshold: float = 0.45):
    """
    Lexical baseline: Computes token overlap (Jaccard similarity) between source and claim.
    If Jaccard similarity >= threshold -> SUPPORTED, else NOT_SUPPORTED.
    """
    def predictor(source: str, claim: str):
        t0 = time.perf_counter_ns()
        s_tokens = set(source.lower().split())
        c_tokens = set(claim.lower().split())
        if not c_tokens:
            jaccard = 0.0
        else:
            intersection = s_tokens.intersection(c_tokens)
            jaccard = len(intersection) / len(c_tokens)
        
        prob = min(1.0, max(0.0, jaccard))
        decision = GroundingLabel.SUPPORTED if jaccard >= threshold else GroundingLabel.NOT_SUPPORTED
        dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)
        return {
            "decision": decision,
            "probability": prob,
            "evidence_spans": [],
            "reason": f"jaccard_overlap_{jaccard:.3f}",
            "latency_us": dt_us
        }
    return predictor


def train_tfidf_logistic_baseline(train_path: str):
    """
    Trains TF-IDF + Logistic Regression on the training split.
    Input formulation: [source] [SEP] [claim]
    """
    train_examples = load_dataset_split(train_path)
    texts = [f"{ex.source} [SEP] {ex.claim}" for ex in train_examples]
    labels = [1 if ex.label == GroundingLabel.SUPPORTED else 0 for ex in train_examples]

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(texts)
    clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
    clf.fit(X_train, labels)

    def predictor(source: str, claim: str):
        t0 = time.perf_counter_ns()
        full_text = f"{source} [SEP] {claim}"
        X = vectorizer.transform([full_text])
        probs = clf.predict_proba(X)[0]
        prob_sup = float(probs[1])
        decision = GroundingLabel.SUPPORTED if prob_sup >= 0.5 else GroundingLabel.NOT_SUPPORTED
        dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)
        return {
            "decision": decision,
            "probability": prob_sup,
            "evidence_spans": [],
            "reason": "tfidf_logistic_inference",
            "latency_us": dt_us
        }

    return predictor


def load_legacy_apexdecision_baseline():
    """
    Loads pre-existing ApexDecision ModernBERT-base with unadapted classification head.
    Evaluates in zero-shot grounding mode: [source] [ASSERTION] [claim].
    """
    import torch
    from apex_decision.src.models.encoder_engine import ApexTransformerEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ApexTransformerEncoder(
        model_name="answerdotai/ModernBERT-base",
        num_classes=8,
        num_levels=5,
        use_lora=False
    )
    # Load previous LoRA if exists
    legacy_path = "apex_decision/build/lora_weights"
    if os.path.exists(legacy_path):
        try:
            model.load_lora(legacy_path)
        except Exception:
            pass
    model.to(device)
    model.eval()

    def predictor(source: str, claim: str):
        t0 = time.perf_counter_ns()
        full_text = f"{source} [ASSERTION] {claim}"
        enc = model.tokenize([full_text], max_length=256)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16 if device == "cuda" else torch.bfloat16):
                logit = model.forward(input_ids, attn_mask, task="binary").item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())

        decision = GroundingLabel.SUPPORTED if prob >= 0.5 else GroundingLabel.NOT_SUPPORTED
        dt_us = max(1, (time.perf_counter_ns() - t0) // 1000)
        return {
            "decision": decision,
            "probability": prob,
            "evidence_spans": [],
            "reason": "legacy_apexdecision_binary",
            "latency_us": dt_us
        }

    return predictor


def run_all_baselines():
    data_dir = "apex_decision/benchmarks/grounding"
    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")
    out_dir = os.path.join(data_dir, "baseline_results")
    os.makedirs(out_dir, exist_ok=True)

    print("==================================================================")
    print("RUNNING BASELINE MODELS ON HELD-OUT TEST SPLIT (600 SAMPLES)")
    print("==================================================================")

    # 1. Lexical Baseline
    print("\n--- 1. Lexical Word-Overlap / Jaccard Baseline ---")
    lexical_pred = build_lexical_baseline(threshold=0.50)
    res_lexical = evaluate_predictor(
        lexical_pred,
        test_path,
        output_results_path=os.path.join(out_dir, "lexical_results.json"),
        model_name="Lexical_Jaccard_Baseline"
    )
    m_lex = res_lexical["metrics"]
    print(f"Lexical Baseline Accuracy: {m_lex['overall_accuracy']*100:.2f}% | Adversarial Acc: {m_lex['adversarial_overall_accuracy']*100:.2f}% | Brier: {m_lex['brier_score']:.4f}")

    # 2. TF-IDF + Logistic Regression
    print("\n--- 2. TF-IDF + Logistic Regression Baseline ---")
    tfidf_pred = train_tfidf_logistic_baseline(train_path)
    res_tfidf = evaluate_predictor(
        tfidf_pred,
        test_path,
        output_results_path=os.path.join(out_dir, "tfidf_logistic_results.json"),
        model_name="TFIDF_Logistic_Baseline"
    )
    m_tfidf = res_tfidf["metrics"]
    print(f"TF-IDF Logistic Accuracy: {m_tfidf['overall_accuracy']*100:.2f}% | Adversarial Acc: {m_tfidf['adversarial_overall_accuracy']*100:.2f}% | Brier: {m_tfidf['brier_score']:.4f}")

    # 3. Legacy ApexDecision Model
    print("\n--- 3. Pre-Existing Legacy ApexDecision Checkpoint ---")
    legacy_pred = load_legacy_apexdecision_baseline()
    res_legacy = evaluate_predictor(
        legacy_pred,
        test_path,
        output_results_path=os.path.join(out_dir, "legacy_apexdecision_results.json"),
        model_name="Legacy_ApexDecision_Baseline"
    )
    m_legacy = res_legacy["metrics"]
    print(f"Legacy ApexDecision Accuracy: {m_legacy['overall_accuracy']*100:.2f}% | Adversarial Acc: {m_legacy['adversarial_overall_accuracy']*100:.2f}% | Brier: {m_legacy['brier_score']:.4f}")

    # Summary table
    print("\n==================================================================")
    print("BASELINE COMPARISON SUMMARY (HELD-OUT TEST SET, N=600)")
    print("==================================================================")
    print(f"{'Baseline System':<32} | {'Accuracy':<10} | {'95% CI':<16} | {'Adversarial Acc':<16} | {'Macro F1':<10} | {'ECE':<8}")
    print("-" * 105)
    for name, m in [
        ("Lexical (Jaccard)", m_lex),
        ("TF-IDF + Logistic", m_tfidf),
        ("Legacy ApexDecision", m_legacy)
    ]:
        ci_str = f"[{m['ci_95'][0]*100:.1f}%, {m['ci_95'][1]*100:.1f}%]" if m.get('ci_95') else "N/A"
        print(f"{name:<32} | {m['overall_accuracy']*100:<9.2f}% | {ci_str:<16} | {m['adversarial_overall_accuracy']*100:<15.2f}% | {m['macro_f1']:<10.4f} | {m['expected_calibration_error']:<8.4f}")
    print("==================================================================")

    # Save summary
    summary_path = os.path.join(out_dir, "baselines_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "lexical": m_lex,
            "tfidf_logistic": m_tfidf,
            "legacy_apexdecision": m_legacy
        }, f, indent=2)
    print(f"Baselines summary saved to {summary_path}")


if __name__ == "__main__":
    run_all_baselines()
