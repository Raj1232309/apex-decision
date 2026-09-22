"""
Run Candidate A: ModernBERT-base + LoRA on Validation and Test Splits.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apex_decision.benchmarks.grounding.dataset_schema import GroundingExample
from apex_decision.benchmarks.grounding_v2.run_bakeoff import profile_model_inference


def run_candidate_a():
    val_path = "apex_decision/benchmarks/grounding/validation.json"
    test_path = "apex_decision/benchmarks/grounding/test.json"
    lora_dir = "apex_decision/build/grounding_lora"

    with open(val_path, "r", encoding="utf-8") as f:
        val_data = [GroundingExample(**item) for item in json.load(f)]
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [GroundingExample(**item) for item in json.load(f)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model = PeftModel.from_pretrained(base_model, lora_dir).to(device)

    print("Evaluating Candidate A (ModernBERT + LoRA) on Validation...")
    val_results = profile_model_inference(
        model, tokenizer, val_data, device, "Candidate_A_ModernBERT_LoRA_Val", is_3class_nli=False
    )

    print("Evaluating Candidate A (ModernBERT + LoRA) on Test...")
    test_results = profile_model_inference(
        model, tokenizer, test_data, device, "Candidate_A_ModernBERT_LoRA_Test", is_3class_nli=False
    )

    out = {
        "candidate_a_validation": val_results,
        "candidate_a_test": test_results
    }
    with open("apex_decision/benchmarks/grounding_v2/candidate_a_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Candidate A results written to candidate_a_results.json")
    return out


if __name__ == "__main__":
    run_candidate_a()
