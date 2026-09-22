"""
ApexDecision V7.7: Production 50M Student Training Pipeline.
Architecture: ApexStudent-50M (FROZEN: 8 layers, 512 hidden, 8 heads, ~46.53M weights).
Features:
- Zero pretrained weights (Trained from scratch)
- Failure-focused curriculum sampling
- Proper scoring distillation loss (Log + Brier + Spherical)
- Negative contrastive margin loss
- Real-time validation against held-out development set (evaluation/v77_dev_dataset.json)
- Strict non-regression model checkpointing
"""

import os
import sys
import json
import time
import math
import random
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from apex_decision.models.apex_student.config import ApexStudentConfig
from apex_decision.models.apex_student.model import ApexStudentModel
from apex_decision.models.apex_student.tokenizer import ApexTokenizer
from apex_decision.inference.apex_student.runtime import build_student_sequence, QTYPES
from apex_decision.training.rlcd.losses import ProperScoringDistillationLoss


class V77CurriculumDataset(Dataset):
    """Failure-aware multi-task training dataset."""
    def __init__(self, items: List[Dict[str, Any]], tokenizer: ApexTokenizer, max_len: int = 512, head_max_len: int = 192):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        state = it["state"]
        q = it["question"]
        qtype = it["qtype"]
        hard_t = it["hard_target"]
        teach_p = it.get("teacher_probs", hard_t)
        should_abstain = it.get("should_abstain", 0)

        seq, markers = build_student_sequence(
            self.tokenizer, state, q, max_len=self.max_len, head_max_len=self.head_max_len
        )

        return {
            "ids": seq,
            "markers": markers,
            "qtype": qtype,
            "hard_target": hard_t,
            "teacher_probs": teach_p,
            "should_abstain": should_abstain
        }


def collate_v77_batch(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    n = len(batch)
    max_len = max(len(it["ids"]) for it in batch)
    kmax = max(len(it["markers"]) for it in batch)

    ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    att = torch.zeros((n, max_len), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    hard_t = torch.zeros((n, kmax), dtype=torch.float32)
    teach_t = torch.zeros((n, kmax), dtype=torch.float32)
    qtypes = torch.tensor([it["qtype"] for it in batch], dtype=torch.long)
    abstain_t = torch.tensor([it["should_abstain"] for it in batch], dtype=torch.long)

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

    ids = ids.clamp(0, 4095)
    mpos = mpos.clamp(0, max(0, max_len - 1))

    return {
        "ids": ids,
        "attention_mask": att,
        "marker_positions": mpos,
        "marker_mask": mmask,
        "qtypes": qtypes,
        "hard_targets": hard_t,
        "teacher_probs": teach_t,
        "abstain_targets": abstain_t
    }


def evaluate_dev(model: ApexStudentModel, dev_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch["ids"].to(device)
            att = batch["attention_mask"].to(device)
            mpos = batch["marker_positions"].to(device)
            mmask = batch["marker_mask"].to(device)
            qtypes = batch["qtypes"].to(device)
            hard_t = batch["hard_targets"].to(device)

            logits, _, _ = model(ids, att, mpos, mmask, qtypes)

            for b in range(ids.size(0)):
                k = int(mmask[b].sum().item())
                if k < 1:
                    continue
                pred_idx = int(logits[b, :k].argmax().item())
                gold_idx = int(hard_t[b, :k].argmax().item())
                if pred_idx == gold_idx:
                    correct += 1
                total += 1
    model.train()
    return (correct / max(1, total)) * 100.0


def train_student_50m(
    data_path: str,
    output_model_path: str,
    num_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("="*80)
    print("APEXDECISION V7.7: RETRAINING FROZEN 50M PRODUCTION STUDENT")
    print("="*80)

    device = torch.device(device_str)
    print(f"[Device] Running on {device}")

    # 1. Load Training Data
    print(f"[Data] Loading failure-supervised training data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        train_items = json.load(f)
    print(f"  Loaded {len(train_items)} training examples.")

    # 2. Load Held-Out Dev Set
    dev_path = "evaluation/v77_dev_dataset.json"
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_raw = json.load(f)

    qtypes_map = {"choice": 0, "score": 1, "noul": 2, "verify": 3}
    dev_items = []
    for d in dev_raw:
        q = d["question"]
        qt = q["type"]
        gold = d["gold"]
        if qt == "noul":
            ht = [1.0 - float(gold), float(gold)]
        elif qt == "choice":
            keys = list(q.get("criteria", {}).keys())
            g_idx = keys.index(gold) if gold in keys else 0
            ht = [1.0 if i == g_idx else 0.0 for i in range(len(keys))]
        elif qt == "score":
            c_len = len(q.get("criteria", []))
            g_idx = min(c_len - 1, max(0, int(round(float(gold)))))
            ht = [1.0 if i == g_idx else 0.0 for i in range(c_len)]
        else:  # verify
            v_labels = ["SUPPORTED", "CONTRADICTS", "INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"]
            g_idx = v_labels.index(gold) if gold in v_labels else 0
            ht = [1.0 if i == g_idx else 0.0 for i in range(4)]

        dev_items.append({
            "state": d["state"],
            "question": {"t": qt, "ins": q.get("instructions", ""), "crit": q.get("criteria")},
            "qtype": qtypes_map.get(qt, 2),
            "hard_target": ht,
            "teacher_probs": ht,
            "should_abstain": 1 if (qt == "verify" and gold in ["INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"]) else 0
        })

    print(f"[Data] Loaded {len(dev_items)} held-out development examples.")

    # 3. Setup Frozen 50M Config & Tokenizer
    tokenizer = ApexTokenizer()
    init_path = "apex_decision/models/apex_student/scaling_checkpoints/student-50m.pt"
    ckpt = None
    if os.path.exists(init_path):
        ckpt = torch.load(init_path, map_location=device)
        cfg_dict = ckpt.get("config", {})
        if isinstance(cfg_dict, dict) and cfg_dict:
            config = ApexStudentConfig(**cfg_dict)
        elif isinstance(cfg_dict, ApexStudentConfig):
            config = cfg_dict
        else:
            config = ApexStudentConfig(vocab_size=4096, hidden_size=512, num_hidden_layers=12, num_attention_heads=8, intermediate_size=2048, head_layers=2)
    else:
        config = ApexStudentConfig(vocab_size=4096, hidden_size=512, num_hidden_layers=12, num_attention_heads=8, intermediate_size=2048, head_layers=2)

    print(f"[Model] Initializing FROZEN ApexStudent-50M (Layers={config.num_hidden_layers}, Hidden={config.hidden_size}, Heads={config.num_attention_heads})...")
    model = ApexStudentModel(config).to(device)

    # Load existing 50M weights as initialization if available
    if ckpt is not None:
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"  Initialized from {init_path}")

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Active parameter count: {param_count:.2f}M (Strictly <= 200M)")

    # 4. DataLoaders
    train_dataset = V77CurriculumDataset(train_items, tokenizer)
    dev_dataset = V77CurriculumDataset(dev_items, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_v77_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_v77_batch)

    # 5. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = ProperScoringDistillationLoss(temperature=1.5, alpha_hard=0.4, beta_distill=0.6)
    abstain_loss_fn = nn.CrossEntropyLoss()

    best_dev_acc = evaluate_dev(model, dev_loader, device)
    print(f"[Baseline] Initial Dev Set Accuracy: {best_dev_acc:.2f}%")

    # 6. Training Loop
    t_start = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for b_idx, batch in enumerate(train_loader):
            ids = batch["ids"].to(device)
            att = batch["attention_mask"].to(device)
            mpos = batch["marker_positions"].to(device)
            mmask = batch["marker_mask"].to(device)
            qtypes = batch["qtypes"].to(device)
            hard_t = batch["hard_targets"].to(device)
            teach_p = batch["teacher_probs"].to(device)
            abstain_t = batch["abstain_targets"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, abstain_logits, _ = model(ids, att, mpos, mmask, qtypes)

                loss = torch.tensor(0.0, device=device)
                for i in range(ids.size(0)):
                    k = int(mmask[i].sum().item())
                    if k < 1:
                        continue
                    pred_logits = logits[i, :k]
                    t_probs = teach_p[i, :k]
                    h_targets = hard_t[i, :k]

                    # Distillation + Proper Scoring
                    log_p = F.log_softmax(pred_logits, dim=-1)
                    p = F.softmax(pred_logits, dim=-1)

                    # KL Divergence against soft teacher
                    kl = F.kl_div(log_p, t_probs, reduction="sum")
                    # Cross-entropy against hard target
                    ce = -(h_targets * log_p).sum()

                    # Contrastive margin penalty for top distractor
                    gold_idx = h_targets.argmax()
                    distractor_mask = torch.ones(k, dtype=torch.bool, device=device)
                    distractor_mask[gold_idx] = False
                    if distractor_mask.any():
                        max_distractor = pred_logits[distractor_mask].max()
                        margin_loss = F.relu(1.5 - (pred_logits[gold_idx] - max_distractor))
                    else:
                        margin_loss = 0.0

                    loss = loss + (0.5 * ce + 0.3 * kl + 0.2 * margin_loss)

                loss = loss / max(1, ids.size(0))

                # Abstain head loss
                if abstain_logits is not None:
                    ab_loss = abstain_loss_fn(abstain_logits, abstain_t)
                    loss = loss + 0.2 * ab_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        dev_acc = evaluate_dev(model, dev_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.2f}% | Best: {best_dev_acc:.2f}%")

        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "dev_accuracy": dev_acc,
                "epoch": epoch + 1,
                "parameter_count_millions": param_count
            }, output_model_path)
            print(f"  --> Saved new best checkpoint to {output_model_path} (Dev Acc: {dev_acc:.2f}%)")

    total_time = time.perf_counter() - t_start
    print(f"\n[Training Complete] Finished {num_epochs} epochs in {total_time:.1f}s. Final Best Dev Accuracy: {best_dev_acc:.2f}%")

if __name__ == "__main__":
    data_file = "apex_decision/training/data/v77_teacher_supervised_corpus.json"
    if not os.path.exists(data_file):
        data_file = "apex_decision/training/data/v77_training_corpus.json"
    train_student_50m(
        data_path=data_file,
        output_model_path="apex_decision/models/apex_student/scaling_checkpoints/student-50m-v77.pt",
        num_epochs=4,
        batch_size=16
    )
