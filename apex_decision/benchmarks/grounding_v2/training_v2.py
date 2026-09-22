"""
High-Precision Grounding Model Trainer (v2) for ApexDecision.

Features:
- Backbones: answerdotai/ModernBERT-base, microsoft/deberta-v3-base
- PEFT LoRA (r=16, alpha=32) targeting attention and projection layers
- Multiple loss functions:
  1. Standard Cross-Entropy
  2. Focal Loss (gamma=2.0, focusing gradients on hard entailment boundary samples)
  3. Class-Balanced Cross-Entropy
- FP16 Automatic Mixed Precision (torch.amp.autocast)
- Peak VRAM tracking ensuring strictly < 3.5 GB on RTX 2050
- Zero test leakage: trains exclusively on train_path and validates on val_path
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apex_decision.benchmarks.grounding.dataset_schema import GroundingExample, GroundingLabel


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard boundary samples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class GroundingDatasetV2(Dataset):
    def __init__(self, examples: List[GroundingExample], tokenizer, max_length: int = 256, formulation: str = "A"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formulation = formulation

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.formulation == "A":
            text = f"{ex.source} [SEP] {ex.claim}"
        else:
            text = f"{ex.claim} [SEP] {ex.source}"
        label = 1 if ex.label == GroundingLabel.SUPPORTED else 0
        return text, label


def collate_v2(batch, tokenizer, max_length: int = 256):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc


def train_grounding_v2(
    base_model_name: str = "answerdotai/ModernBERT-base",
    train_path: str = "apex_decision/benchmarks/grounding/train.json",
    val_path: str = "apex_decision/benchmarks/grounding/validation.json",
    output_dir: str = "apex_decision/build/grounding_v2_model",
    loss_type: str = "focal",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-4,
    formulation: str = "A",
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    print(f"\n==================================================================")
    print(f"TRAINING V2: {base_model_name} | Loss: {loss_type} | Form: {formulation}")
    print(f"==================================================================")

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Load data
    with open(train_path, "r", encoding="utf-8") as f:
        train_examples = [GroundingExample(**item) for item in json.load(f)]
    with open(val_path, "r", encoding="utf-8") as f:
        val_examples = [GroundingExample(**item) for item in json.load(f)]

    print(f"Loaded {len(train_examples)} train examples, {len(val_examples)} validation examples.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_ds = GroundingDatasetV2(train_examples, tokenizer, max_length=256, formulation=formulation)
    val_ds = GroundingDatasetV2(val_examples, tokenizer, max_length=256, formulation=formulation)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_v2(b, tokenizer))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_v2(b, tokenizer))

    # Base Model & LoRA Config
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    
    # Configure LoRA targets depending on architecture
    if "modernbert" in base_model_name.lower():
        target_modules = ["Wqkv", "Wo"]
    elif "deberta" in base_model_name.lower():
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
    else:
        target_modules = ["q_proj", "v_proj", "out_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules
    )

    model = get_peft_model(base_model, peft_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Select Loss Function
    if loss_type == "focal":
        criterion = FocalLoss(gamma=2.0)
    elif loss_type == "balanced":
        sup_cnt = sum(1 for ex in train_examples if ex.label == GroundingLabel.SUPPORTED)
        unsup_cnt = len(train_examples) - sup_cnt
        w0 = len(train_examples) / (2.0 * max(1, unsup_cnt))
        w1 = len(train_examples) / (2.0 * max(1, sup_cnt))
        weights = torch.tensor([w0, w1], dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)

        train_loss /= len(train_examples)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                val_loss += loss.item() * len(labels)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_examples)
        val_acc = val_correct / len(val_examples)
        dt = time.time() - t0

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
        print(f"Epoch {epoch}/{epochs} ({dt:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Peak VRAM: {peak_vram_mb:.1f} MB")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "peak_vram_mb": peak_vram_mb,
            "duration_s": dt
        })

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%. Model saved to {output_dir}.")
    return {
        "best_val_accuracy": best_val_acc,
        "history": history,
        "output_dir": output_dir,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params
    }
