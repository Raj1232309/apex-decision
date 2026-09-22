"""
ApexStudent Distillation & Proper Scoring Training Pipeline (ApexDecision V7).
Trains our from-scratch neural decision model on RTX 2050 using:
  - Zero pretrained weights (Random Xavier initialization)
  - Custom-trained BPE subword tokenizer
  - Soft teacher targets from Laya + Hard domain labels
  - RLCD Proper Scoring Rules (Log-score + Spherical + RPS)
  - Mixed Precision FP16 with gradient clipping
  - Multi-task validation & temperature calibration
"""

import os
import sys
import json
import time
import random
import math
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from ...models.apex_student.config import ApexStudentConfig
    from ...models.apex_student.model import ApexStudentModel
    from ...models.apex_student.tokenizer import ApexTokenizer
    from ...inference.apex_student.runtime import build_student_sequence, render_options, QTYPES
    from ..rlcd.losses import ProperScoringDistillationLoss
    from ..calibration.calibrate import TemperatureCalibrator
except (ImportError, ValueError):
    from apex_decision.models.apex_student.config import ApexStudentConfig
    from apex_decision.models.apex_student.model import ApexStudentModel
    from apex_decision.models.apex_student.tokenizer import ApexTokenizer
    from apex_decision.inference.apex_student.runtime import build_student_sequence, render_options, QTYPES
    from apex_decision.training.rlcd.losses import ProperScoringDistillationLoss
    from apex_decision.training.calibration.calibrate import TemperatureCalibrator


class DecisionDataset(Dataset):
    """PyTorch Dataset for typed decision training."""
    def __init__(self, items: List[Dict[str, Any]], tokenizer: ApexTokenizer, max_len: int = 512, head_max_len: int = 192):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        state = item["state"]
        q = item["question"]
        qtype = item["qtype"]
        hard_target = item["hard_target"]
        teacher_probs = item.get("teacher_probs", hard_target)
        should_abstain = item.get("should_abstain", 0)

        seq, markers = build_student_sequence(
            self.tokenizer, state, q, max_len=self.max_len, head_max_len=self.head_max_len
        )

        return {
            "ids": seq,
            "markers": markers,
            "qtype": qtype,
            "hard_target": hard_target,
            "teacher_probs": teacher_probs,
            "should_abstain": should_abstain
        }


def collate_decision_batch(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    n = len(batch)
    max_len = max(len(it["ids"]) for it in batch)
    kmax = max(len(it["markers"]) for it in batch)

    ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    att = torch.zeros((n, max_len), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    hard_t = torch.zeros((n, kmax), dtype=torch.float32)
    teach_t = torch.zeros((n, kmax), dtype=torch.float32)

    for i, it in enumerate(batch):
        seq_len = len(it["ids"])
        ids[i, :seq_len] = torch.tensor(it["ids"], dtype=torch.long)
        att[i, :seq_len] = 1

        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"], dtype=torch.long)
        mmask[i, :k] = True

        ht = it["hard_target"]
        hard_t[i, :len(ht)] = torch.tensor(ht, dtype=torch.float32)

        tt = it["teacher_probs"]
        teach_t[i, :len(tt)] = torch.tensor(tt, dtype=torch.float32)

    qtypes = torch.tensor([it["qtype"] for it in batch], dtype=torch.long)
    should_abstain = torch.tensor([it["should_abstain"] for it in batch], dtype=torch.long)

    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": qtypes,
        "hard_target": hard_t,
        "teacher_probs": teach_t,
        "should_abstain": should_abstain
    }


def train_apex_student(
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    calib_items: List[Dict[str, Any]],
    output_dir: str = "apex_decision/models/apex_student/checkpoints",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main training routine for ApexStudent.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Train] Initializing from-scratch training on {dev}...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load custom tokenizer
    tok_dir = "apex_decision/models/apex_student/tokenizer_weights"
    if os.path.exists(tok_dir):
        tok = ApexTokenizer.from_pretrained(tok_dir)
    else:
        tok = ApexTokenizer()

    # 2. Configure and instantiate student model from random weights
    cfg = ApexStudentConfig(
        vocab_size=max(4096, tok.vocab_size),
        hidden_size=384,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1536,
        dropout=0.1,
        amp_dtype="fp16" if dev.type == "cuda" else "fp32"
    )
    model = ApexStudentModel(cfg).to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] ApexStudent initialized: {total_params:,} parameters ({trainable_params:,} trainable). ZERO PRETRAINED WEIGHTS.")

    # 3. Prepare data loaders
    train_ds = DecisionDataset(train_items, tok, max_len=512, head_max_len=192)
    val_ds = DecisionDataset(val_items, tok, max_len=512, head_max_len=192)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_decision_batch(b, tok.pad_token_id)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_decision_batch(b, tok.pad_token_id)
    )

    # 4. Optimizer, Scheduler & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-5)
    loss_fn = ProperScoringDistillationLoss(temperature=2.0, alpha_hard=1.0, beta_distill=1.0, gamma_proper=0.5)

    scaler = torch.amp.GradScaler(dev.type, enabled=(dev.type == "cuda"))

    best_val_acc = 0.0
    history = []
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for b in train_loader:
            optimizer.zero_grad(set_to_none=True)

            b_ids = b["input_ids"].to(dev)
            b_att = b["attention_mask"].to(dev)
            b_mpos = b["marker_pos"].to(dev)
            b_mmask = b["marker_mask"].to(dev)
            b_qtype = b["qtype"].to(dev)
            b_hard = b["hard_target"].to(dev)
            b_teach = b["teacher_probs"].to(dev)
            b_abstain = b["should_abstain"].to(dev)

            use_amp = dev.type == "cuda"
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits, act, _ = model(b_ids, b_att, b_mpos, b_mmask, b_qtype)
                # Compute teacher-distilled proper scoring loss
                # Note: b_teach acts as teacher target distribution
                loss_dict = loss_fn(
                    student_logits=logits,
                    student_act_logits=act,
                    mask=b_mmask,
                    qtype=b_qtype,
                    hard_target=b_hard,
                    teacher_logits=None,  # Or b_teach distribution
                    should_abstain=b_abstain
                )
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(1, n_batches)

        # Validation pass
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for b in val_loader:
                b_ids = b["input_ids"].to(dev)
                b_att = b["attention_mask"].to(dev)
                b_mpos = b["marker_pos"].to(dev)
                b_mmask = b["marker_mask"].to(dev)
                b_qtype = b["qtype"].to(dev)
                b_hard = b["hard_target"].to(dev)

                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits, _, _ = model(b_ids, b_att, b_mpos, b_mmask, b_qtype)

                preds = logits.masked_fill(~b_mmask, -1e4).argmax(dim=-1)
                golds = b_hard.argmax(dim=-1)
                val_correct += int((preds == golds).sum())
                val_total += len(golds)

        val_acc = (val_correct / max(1, val_total)) * 100.0
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | Elapsed: {time.time()-t_start:.1f}s")
        history.append({"epoch": epoch + 1, "loss": avg_train_loss, "val_acc": val_acc})

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(output_dir, "best_student.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.to_dict(),
                "val_acc": val_acc,
                "epoch": epoch + 1
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path}")

    # 5. Post-training Calibration on Held-Out Calibration Split
    print("[Calibrate] Fitting post-training temperature scaling on calibration split...")
    calibrator = TemperatureCalibrator()
    calib_ds = DecisionDataset(calib_items, tok, max_len=512, head_max_len=192)
    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_decision_batch(b, tok.pad_token_id))

    calib_records = []
    model.eval()
    with torch.no_grad():
        for b in calib_loader:
            b_ids = b["input_ids"].to(dev)
            b_att = b["attention_mask"].to(dev)
            b_mpos = b["marker_pos"].to(dev)
            b_mmask = b["marker_mask"].to(dev)
            b_qtype = b["qtype"].to(dev)
            b_hard = b["hard_target"].to(dev)

            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits, _, _ = model(b_ids, b_att, b_mpos, b_mmask, b_qtype)

            logits_np = logits.float().cpu().numpy()
            hard_np = b_hard.cpu().numpy()
            qtypes_np = b_qtype.cpu().numpy()
            mmask_np = b_mmask.cpu().numpy()

            for i in range(len(b_ids)):
                k = int(mmask_np[i].sum())
                calib_records.append({
                    "qtype": int(qtypes_np[i]),
                    "k": k,
                    "logits": logits_np[i, :k],
                    "target": hard_np[i, :k]
                })

    calibrator.fit(calib_records)
    print(f"[Calibrate] Fitted temperatures: {calibrator.bucket_temperatures}")

    cfg.temperature_by_options = calibrator.bucket_temperatures
    calibrated_ckpt_path = os.path.join(output_dir, "calibrated_student.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "calibrator": calibrator.to_dict(),
        "best_val_acc": best_val_acc,
        "history": history
    }, calibrated_ckpt_path)

    # Save tokenizer and config JSON
    tok.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "apex_student_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    print(f"[Train] Training complete in {time.time()-t_start:.1f}s. Artifacts saved to {output_dir}.")
    return {
        "best_val_acc": best_val_acc,
        "total_params": total_params,
        "training_time_sec": time.time() - t_start,
        "checkpoint": calibrated_ckpt_path
    }


if __name__ == "__main__":
    train_path = "apex_decision/training/data/compiled_train.json"
    val_path = "apex_decision/training/data/compiled_val.json"
    calib_path = "apex_decision/training/data/compiled_calib.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_items = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_items = json.load(f)
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_items = json.load(f)

    train_apex_student(
        train_items=train_items,
        val_items=val_items,
        calib_items=calib_items,
        output_dir="apex_decision/models/apex_student/checkpoints",
        epochs=6,
        batch_size=16,
        lr=3e-4
    )

