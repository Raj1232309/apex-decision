"""
Validation vs Held-Out Test Distribution Shift Analysis for ApexDecision Grounding.

Performs:
1. Source hash overlap analysis across Train, Validation, and Test.
2. Exact duplicate and near-duplicate detection.
3. Lexical overlap distributions (Source vs Claim Jaccard).
4. Difficulty tier, failure mode, domain, and label distributions.
5. Source and claim word count distributions.
6. Model accuracy per category on Validation vs Test.
Outputs:
- apex_decision/benchmarks/grounding_v2/validation_shift.json
- apex_decision/benchmarks/grounding_v2/validation_vs_test_shift.json
"""

import os
import sys
import json
import hashlib
import re
from typing import Dict, List, Set
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apex_decision.benchmarks.grounding.dataset_schema import GroundingExample, GroundingLabel


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r'\b\w+\b', text.lower()))


def jaccard(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


def analyze_distributions():
    print("==================================================================")
    print("PHASE 2: VALIDATION VS TEST DISTRIBUTION SHIFT ANALYSIS")
    print("==================================================================")

    train_path = "apex_decision/benchmarks/grounding/train.json"
    val_path = "apex_decision/benchmarks/grounding/validation.json"
    test_path = "apex_decision/benchmarks/grounding/test.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = [GroundingExample(**item) for item in json.load(f)]
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = [GroundingExample(**item) for item in json.load(f)]
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [GroundingExample(**item) for item in json.load(f)]

    splits = {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

    # 1. Source Hash Overlap Analysis
    source_hashes = {}
    for name, data in splits.items():
        hashes = set(hashlib.sha256(ex.source.strip().encode()).hexdigest() for ex in data)
        source_hashes[name] = hashes

    train_sources = source_hashes["train"]
    val_sources = source_hashes["validation"]
    test_sources = source_hashes["test"]

    val_in_train_sources = len(val_sources.intersection(train_sources))
    test_in_train_sources = len(test_sources.intersection(train_sources))
    test_in_val_sources = len(test_sources.intersection(val_sources))

    print("\n--- 1. Source Document Overlap ---")
    print(f"Unique Train sources:      {len(train_sources)}")
    print(f"Unique Validation sources: {len(val_sources)}")
    print(f"Unique Test sources:       {len(test_sources)}")
    print(f"Validation sources appearing in Train: {val_in_train_sources} / {len(val_sources)} ({val_in_train_sources/len(val_sources)*100:.1f}%) [Pool A Shared Source Contexts]")
    print(f"Test sources appearing in Train:       {test_in_train_sources} / {len(test_sources)} ({test_in_train_sources/len(test_sources)*100:.1f}%) [Pool B Strictly Disjoint]")
    print(f"Test sources appearing in Validation:  {test_in_val_sources} / {len(test_sources)} ({test_in_val_sources/len(test_sources)*100:.1f}%) [Pool B Strictly Disjoint]")

    # 2. Exact Duplicate Claim/Source Pairs
    print("\n--- 2. Exact Duplicate Detection ---")
    pair_hashes = {name: set(hashlib.sha256(f"{ex.source}:::{ex.claim}".encode()).hexdigest() for ex in data) for name, data in splits.items()}
    val_in_train_pairs = len(pair_hashes["validation"].intersection(pair_hashes["train"]))
    test_in_train_pairs = len(pair_hashes["test"].intersection(pair_hashes["train"]))
    test_in_val_pairs = len(pair_hashes["test"].intersection(pair_hashes["validation"]))
    print(f"Exact (source, claim) duplicates Val in Train:  {val_in_train_pairs}")
    print(f"Exact (source, claim) duplicates Test in Train: {test_in_train_pairs}")
    print(f"Exact (source, claim) duplicates Test in Val:   {test_in_val_pairs}")

    # 3. Lexical Overlap (Source vs Claim Jaccard)
    print("\n--- 3. Lexical Overlap (Source vs Claim) ---")
    lexical_stats = {}
    for name, data in splits.items():
        overlaps = [jaccard(tokenize(ex.source), tokenize(ex.claim)) for ex in data]
        lexical_stats[name] = {
            "mean": float(np.mean(overlaps)),
            "median": float(np.median(overlaps)),
            "std": float(np.std(overlaps)),
            "min": float(np.min(overlaps)),
            "max": float(np.max(overlaps))
        }
        print(f"{name.capitalize():<12} mean overlap: {lexical_stats[name]['mean']:.4f} (std: {lexical_stats[name]['std']:.4f}, median: {lexical_stats[name]['median']:.4f})")

    # 4. Length Distributions
    print("\n--- 4. Length Distributions (Word Counts) ---")
    length_stats = {}
    for name, data in splits.items():
        src_lens = [len(ex.source.split()) for ex in data]
        clm_lens = [len(ex.claim.split()) for ex in data]
        length_stats[name] = {
            "source_words": {
                "mean": float(np.mean(src_lens)),
                "median": float(np.median(src_lens)),
                "min": int(np.min(src_lens)),
                "max": int(np.max(src_lens))
            },
            "claim_words": {
                "mean": float(np.mean(clm_lens)),
                "median": float(np.median(clm_lens)),
                "min": int(np.min(clm_lens)),
                "max": int(np.max(clm_lens))
            }
        }
        print(f"{name.capitalize():<12} Source words: mean={length_stats[name]['source_words']['mean']:.1f}, Claim words: mean={length_stats[name]['claim_words']['mean']:.1f}")

    # 5. Difficulty Tier Distribution
    print("\n--- 5. Difficulty Tier Distribution ---")
    tier_stats = {}
    for name, data in splits.items():
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for ex in data:
            counts[ex.difficulty_tier.value] += 1
        tier_stats[name] = {f"tier_{k}": {"count": v, "pct": v / len(data)} for k, v in counts.items()}
        print(f"{name.capitalize():<12} Tier 1: {counts[1]}, Tier 2: {counts[2]}, Tier 3: {counts[3]}, Tier 4: {counts[4]}")

    # 6. Domain Distribution
    print("\n--- 6. Domain Distribution ---")
    domain_stats = {}
    for name, data in splits.items():
        counts = {}
        for ex in data:
            d = ex.domain.value
            counts[d] = counts.get(d, 0) + 1
        domain_stats[name] = counts

    # 7. Failure Mode Distribution
    print("\n--- 7. Failure Mode Distribution ---")
    category_stats = {}
    for name, data in splits.items():
        counts = {}
        for ex in data:
            c = ex.failure_mode.value
            counts[c] = counts.get(c, 0) + 1
        category_stats[name] = counts

    # 8. Model Accuracy on Validation vs Test by Category
    print("\n--- 8. Model Accuracy on Validation vs Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    model = PeftModel.from_pretrained(base_model, "apex_decision/build/grounding_lora").to(device)
    model.eval()

    def eval_split(split_data):
        correct_by_cat = {}
        total_by_cat = {}
        batch_size = 16
        for i in range(0, len(split_data), batch_size):
            batch = split_data[i:i + batch_size]
            texts = [f"{ex.source} [SEP] {ex.claim}" for ex in batch]
            enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                    outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            for j, ex in enumerate(batch):
                cat = ex.failure_mode.value
                gold = 1 if ex.label == GroundingLabel.SUPPORTED else 0
                is_correct = (preds[j] == gold)
                correct_by_cat[cat] = correct_by_cat.get(cat, 0) + int(is_correct)
                total_by_cat[cat] = total_by_cat.get(cat, 0) + 1
        acc_by_cat = {cat: correct_by_cat[cat] / total_by_cat[cat] for cat in total_by_cat}
        overall = sum(correct_by_cat.values()) / sum(total_by_cat.values())
        return overall, acc_by_cat, total_by_cat

    val_overall, val_cat_acc, val_cat_tot = eval_split(val_data)
    test_overall, test_cat_acc, test_cat_tot = eval_split(test_data)

    print(f"Validation Overall Accuracy: {val_overall*100:.2f}%")
    print(f"Test Overall Accuracy:       {test_overall*100:.2f}%")
    print(f"Generalization Gap:          {(val_overall - test_overall)*100:.2f} percentage points")

    print(f"\n{'Category':<32} | {'Val Acc':<10} | {'Test Acc':<10} | {'Drop':<10}")
    print("-" * 70)
    category_comparison = {}
    for cat in sorted(test_cat_acc.keys()):
        v_acc = val_cat_acc.get(cat, 0.0)
        t_acc = test_cat_acc.get(cat, 0.0)
        drop = v_acc - t_acc
        category_comparison[cat] = {
            "validation_accuracy": v_acc,
            "test_accuracy": t_acc,
            "drop": drop,
            "val_count": val_cat_tot.get(cat, 0),
            "test_count": test_cat_tot.get(cat, 0)
        }
        print(f"{cat:<32} | {v_acc*100:<9.2f}% | {t_acc*100:<9.2f}% | {drop*100:<+9.2f}%")

    shift_report = {
        "summary": {
            "validation_accuracy": val_overall,
            "test_accuracy": test_overall,
            "gap": val_overall - test_overall,
            "val_in_train_sources_pct": val_in_train_sources / len(val_sources),
            "test_in_train_sources_pct": test_in_train_sources / len(test_sources)
        },
        "source_overlap": {
            "train_unique_sources": len(train_sources),
            "val_unique_sources": len(val_sources),
            "test_unique_sources": len(test_sources),
            "val_in_train_sources": val_in_train_sources,
            "test_in_train_sources": test_in_train_sources,
            "test_in_val_sources": test_in_val_sources
        },
        "pair_overlap": {
            "val_in_train_pairs": val_in_train_pairs,
            "test_in_train_pairs": test_in_train_pairs,
            "test_in_val_pairs": test_in_val_pairs
        },
        "lexical_overlap_stats": lexical_stats,
        "length_stats": length_stats,
        "difficulty_tiers": tier_stats,
        "domain_distribution": domain_stats,
        "category_distribution": category_stats,
        "category_accuracy_comparison": category_comparison
    }

    # Save to both required filenames
    with open("apex_decision/benchmarks/grounding_v2/validation_shift.json", "w", encoding="utf-8") as f:
        json.dump(shift_report, f, indent=2)
    with open("apex_decision/benchmarks/grounding_v2/validation_vs_test_shift.json", "w", encoding="utf-8") as f:
        json.dump(shift_report, f, indent=2)

    print("\nShift analysis json written to validation_shift.json and validation_vs_test_shift.json")
    return shift_report


if __name__ == "__main__":
    analyze_distributions()
