"""
Knowledge Distillation Module for ApexDecision Grounding.

Transfers representation knowledge from a teacher NLI model (e.g. NLI-DeBERTa-v3)
to a student encoder (e.g. ModernBERT + LoRA).

Loss:
L = (1 - alpha) * L_CE(y_student, y_hard) + alpha * (T^2) * KL_Div(y_student / T, y_teacher / T)
"""

import os
import time
import json
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


class DistillationLoss(nn.Module):
    """
    Combined Hard Label (Cross-Entropy) and Soft Teacher Target (KL Divergence) Loss.
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        hard_labels: torch.Tensor
    ) -> torch.Tensor:
        # 1. Hard label loss
        loss_ce = self.ce_loss(student_logits, hard_labels)

        # 2. Soft distillation loss
        # If student has 2 classes and teacher has 3 classes, collapse teacher [contra, neutral] -> class 0
        if student_logits.shape[-1] == 2 and teacher_probs.shape[-1] == 3:
            collapsed_teacher = torch.stack([
                teacher_probs[:, 0] + teacher_probs[:, 2], # Not supported
                teacher_probs[:, 1]                        # Supported
            ], dim=-1)
            target_probs = collapsed_teacher
        else:
            target_probs = teacher_probs

        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        loss_kd = self.kl_div(student_log_probs, target_probs) * (self.temperature ** 2)

        return (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd


class TeacherDistiller:
    """
    Generates teacher soft probability distributions over training datasets.
    """
    def __init__(
        self,
        teacher_model_path: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(teacher_model_path).to(self.device)
        self.model.eval()

    def generate_soft_labels(
        self,
        sources: List[str],
        claims: List[str],
        batch_size: int = 16
    ) -> np.ndarray:
        all_probs = []
        for i in range(0, len(sources), batch_size):
            src_batch = sources[i:i + batch_size]
            clm_batch = claims[i:i + batch_size]

            enc = self.tokenizer(
                src_batch,
                clm_batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=torch.float16):
                    logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)
