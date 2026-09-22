"""
Hard-Negative Mining for ApexDecision Grounding.

Performs:
1. Full-pass inference over train.json using the trained LoRA checkpoint.
2. Identifies high-confidence false positives and false negatives (loss > threshold or margin > 0.5).
3. Synthesizes an augmented hard-negative training set: train_augmented.json.
4. Strictly leaves the test set isolated.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from .dataset_schema import GroundingExample, GroundingLabel


def mine_hard_negatives(
    model_dir: str = "apex_decision/build/grounding_lora",
    train_path: str = "apex_decision/benchmarks/grounding/train.json",
    output_path: str = "apex_decision/benchmarks/grounding/train_augmented.json",
    confidence_margin: float = 0.35,
    boost_replications: int = 2
):
    print(f"Starting Hard-Negative Mining on {train_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = [GroundingExample(**item) for item in data]

    # Load model
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    model.eval()

    hard_examples = []
    regular_examples = []

    batch_size = 16
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        texts = [f"{ex.source} [SEP] {ex.claim}" for ex in batch]
        labels = [1 if ex.label == GroundingLabel.SUPPORTED else 0 for ex in batch]

        enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for j, ex in enumerate(batch):
            prob_sup = float(probs[j, 1])
            true_label = labels[j]

            # Error or low margin:
            pred_label = 1 if prob_sup >= 0.5 else 0
            is_error = (pred_label != true_label)
            is_uncertain = abs(prob_sup - 0.5) < confidence_margin

            if is_error or is_uncertain:
                hard_examples.append(ex)
            else:
                regular_examples.append(ex)

    print(f"Mined {len(hard_examples)} hard-negative / boundary examples out of {len(examples)} training items.")

    # Replicate hard examples to reweight loss towards boundary cases
    augmented_list = list(examples)
    for _ in range(boost_replications):
        augmented_list.extend(hard_examples)

    print(f"Total augmented dataset size: {len(augmented_list)} examples.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([ex.model_dump(mode="json") for ex in augmented_list], f, indent=2)

    print(f"Augmented dataset saved to {output_path}")
    return output_path


if __name__ == "__main__":
    mine_hard_negatives()
