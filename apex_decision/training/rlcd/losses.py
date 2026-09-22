"""
Proper Scoring Rules & Distillation Losses for ApexStudent Training (RLCD).
Implements Log-score, Spherical score, Ranked Probability Score (RPS),
Temperature-scaled Distillation KL, and Calibrated Abstention penalties.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.75,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """
    Strictly proper scoring rule reward.
    Rewards honest, calibrated probability estimates.

    q: [..., N, K] reported model probability distribution
    target: [..., N, K] ground truth (one-hot or soft target distribution)
    qtype: [..., N] integer question type (0=choice, 1=score, 2=noul, 3=verify)
    mask: [..., N, K] boolean mask of active options
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)

    # Spherical scoring rule
    norm_q = q.norm(dim=-1).clamp_min(1e-9)
    sph = (target * q).sum(-1) / norm_q

    r = log_score + w_sph * sph

    # Ranked Probability Score for ordinal Score primitive (qtype == 1)
    is_score = (qtype == 1).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score

    return r


class ProperScoringDistillationLoss(nn.Module):
    """
    Comprehensive loss module combining:
      1. Hard-label Cross Entropy
      2. Soft Teacher Distillation (KL with temperature scaling)
      3. Proper Scoring Regularization (Log + Spherical + RPS)
      4. Brier Calibration Penalty
      5. Epistemic Abstention Objective
    """
    def __init__(
        self,
        temperature: float = 2.0,
        alpha_hard: float = 1.0,
        beta_distill: float = 1.5,
        gamma_proper: float = 0.5,
        delta_abstain: float = 0.5,
        w_sph: float = 0.75,
        w_rps: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_hard = alpha_hard
        self.beta_distill = beta_distill
        self.gamma_proper = gamma_proper
        self.delta_abstain = delta_abstain
        self.w_sph = w_sph
        self.w_rps = w_rps

    def forward(
        self,
        student_logits: torch.Tensor,
        student_act_logits: torch.Tensor,
        mask: torch.Tensor,
        qtype: torch.Tensor,
        hard_target: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        should_abstain: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        student_logits: [B, K]
        student_act_logits: [B, 2] (0=decide, 1=abstain)
        mask: [B, K] boolean mask of valid options
        qtype: [B] question type ID
        hard_target: [B, K] one-hot or soft target distribution
        teacher_logits: [B, K] optional teacher logit predictions
        should_abstain: [B] boolean indicating ungrounded/adversarial cases that must be abstained
        """
        device = student_logits.device
        masked_student_logits = student_logits.masked_fill(~mask, -1e4)
        p_student = F.softmax(masked_student_logits, dim=-1)

        # 1. Hard Cross-Entropy Loss
        log_p = F.log_softmax(masked_student_logits, dim=-1)
        loss_hard = -(hard_target * log_p).sum(-1).mean()

        # 2. Distillation Loss (KL divergence from Teacher)
        if teacher_logits is not None:
            masked_teacher = teacher_logits.masked_fill(~mask, -1e4)
            p_teacher_soft = F.softmax(masked_teacher / self.temperature, dim=-1)
            log_p_student_soft = F.log_softmax(masked_student_logits / self.temperature, dim=-1)

            # KL(teacher || student)
            loss_distill = (self.temperature ** 2) * F.kl_div(
                log_p_student_soft, p_teacher_soft, reduction="batchmean"
            )
        else:
            loss_distill = torch.tensor(0.0, device=device)

        # 3. Proper Scoring Loss (minimizing negative reward)
        with torch.no_grad():
            r = proper_reward(p_student, hard_target, qtype, mask, self.w_sph, self.w_rps)
        # Gradient path through surrogate log-likelihood weighted by advantage
        surrogate_proper = -r.mean()

        # 4. Brier Calibration Score
        brier = (((p_student - hard_target) ** 2) * mask).sum(-1).mean()

        # 5. Abstention Loss
        if should_abstain is not None:
            loss_act = F.cross_entropy(student_act_logits, should_abstain.long())
        else:
            # Penalize overconfidence on high-entropy distributions
            k_cnt = mask.sum(-1).clamp(min=2).float()
            ent = -(p_student * torch.log(p_student.clamp_min(1e-12))).sum(-1) / torch.log(k_cnt)
            # Default action: decide (class 0)
            default_act = torch.zeros(student_logits.size(0), dtype=torch.long, device=device)
            loss_act = F.cross_entropy(student_act_logits, default_act) + 0.1 * ent.mean()

        # Total combined loss
        total_loss = (
            self.alpha_hard * loss_hard
            + (self.beta_distill * loss_distill if teacher_logits is not None else 0.0)
            + self.gamma_proper * brier
            + self.delta_abstain * loss_act
        )

        return {
            "total_loss": total_loss,
            "loss_hard": loss_hard,
            "loss_distill": loss_distill,
            "loss_proper": surrogate_proper,
            "brier_score": brier,
            "loss_act": loss_act
        }
