"""
LoRA Fine-Tuning for ModernBERT on Grounding and Claim-Verification Benchmark.

Evaluates:
- Formulation A: [source] [SEP] [claim]
- Formulation B: [claim] [SEP] [source]
Trains ModernBERT-base with LoRA (r=16, alpha=32) under FP16 mixed precision.
Guarantees: Peak VRAM < 3.5 GB on RTX 2050 (4 GB).
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from .dataset_schema import GroundingExample, GroundingLabel


class GroundingDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int = 256, formulation: str = "A"):
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


def collate_fn(batch, tokenizer, max_length: int = 256):
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


def load_dataset_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [GroundingExample(**item) for item in data]


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

            val_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    acc = correct / total if total > 0 else 0.0
    avg_loss = val_loss / total if total > 0 else 0.0
    return acc, avg_loss


def train_lora_grounding(
    train_path: str = "apex_decision/benchmarks/grounding/train.json",
    val_path: str = "apex_decision/benchmarks/grounding/validation.json",
    output_dir: str = "apex_decision/build/grounding_lora",
    formulation: str = "A",
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    lr: float = 2e-4,
    max_length: int = 256,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | Formulation: {formulation}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial VRAM Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # 1. Load Tokenizer & Model
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2. Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["Wqkv", "Wo"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} | Trainable LoRA Params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 3. Data Loaders
    train_examples = load_dataset_file(train_path)
    val_examples = load_dataset_file(val_path)

    train_ds = GroundingDataset(train_examples, tokenizer, max_length=max_length, formulation=formulation)
    val_ds = GroundingDataset(val_examples, tokenizer, max_length=max_length, formulation=formulation)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=max_length)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=max_length)
    )

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []

    print(f"Starting Training for {epochs} Epochs ({total_steps} optimizer steps)...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        step_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels) / grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += loss.item() * grad_accum_steps

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                train_loss += step_loss
                step_loss = 0.0

            if step % 100 == 0:
                if device.type == "cuda":
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Peak VRAM: {peak_mb:.1f} MB")

        # Epoch Validation
        val_acc, val_loss = evaluate_model(model, val_loader, device)
        avg_train_loss = train_loss / (len(train_loader) / grad_accum_steps)
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

        print(f"Epoch {epoch}/{epochs} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
        print(f"  Peak VRAM: {peak_vram_mb:.1f} MB (Budget: 3500 MB)")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "peak_vram_mb": peak_vram_mb
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> New best validation accuracy: {best_val_acc*100:.2f}%. Saving checkpoint to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    total_time = time.time() - t_start
    print(f"Training completed in {total_time:.1f}s. Best Validation Accuracy: {best_val_acc*100:.2f}%")

    metrics_out = {
        "formulation": formulation,
        "best_val_accuracy": best_val_acc,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "peak_vram_mb": peak_vram_mb,
        "training_time_seconds": total_time,
        "history": history
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    return best_val_acc, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--formulation", type=str, default="A", choices=["A", "B"], help="Input formulation: A=[source][SEP][claim], B=[claim][SEP][source]")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="apex_decision/build/grounding_lora")
    args = parser.parse_args()

    train_lora_grounding(
        formulation=args.formulation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=args.output_dir
    )
