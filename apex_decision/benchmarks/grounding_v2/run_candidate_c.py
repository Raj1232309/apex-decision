"""
Run Candidate C: cross-encoder/nli-deberta-v3-base adapted to Grounding.

Documents 3-Class NLI to Grounding Mapping:
- entailment: P(SUPPORTED) = P(entailment)
- contradiction: P(NOT_SUPPORTED_refuted) = P(contradiction)
- neutral: P(NOT_SUPPORTED_insufficient) = P(neutral)
Decision: SUPPORTED iff P(entailment) >= 0.50.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apex_decision.benchmarks.grounding.dataset_schema import GroundingExample
from apex_decision.benchmarks.grounding_v2.run_bakeoff import profile_model_inference


def run_candidate_c():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    val_path = os.path.join(base_dir, "grounding", "validation.json")
    test_path = os.path.join(base_dir, "grounding", "test.json")
    out_path = os.path.join(base_dir, "grounding_v2", "candidate_c_results.json")
    snapshot_dir = os.getenv("APEX_BENCHMARK_MODEL_PATH", "")
    model_name = snapshot_dir if os.path.exists(snapshot_dir) else "cross-encoder/nli-deberta-v3-base"

    with open(val_path, "r", encoding="utf-8") as f:
        val_data = [GroundingExample(**item) for item in json.load(f)]
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [GroundingExample(**item) for item in json.load(f)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True).to(device)

    print("Evaluating Candidate C (NLI-DeBERTa-v3) on Validation...")
    val_results = profile_model_inference(
        model, tokenizer, val_data, device, "Candidate_C_NLI_DeBERTa_Val", is_3class_nli=True
    )

    print("Evaluating Candidate C (NLI-DeBERTa-v3) on Test...")
    test_results = profile_model_inference(
        model, tokenizer, test_data, device, "Candidate_C_NLI_DeBERTa_Test", is_3class_nli=True
    )

    out = {
        "candidate_c_validation": val_results,
        "candidate_c_test": test_results
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Candidate C results written to {out_path}")
    return out


if __name__ == "__main__":
    run_candidate_c()
