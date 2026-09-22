"""
ApexStudentModel: Independent from-scratch neural decision model for ApexDecision V7.
Constructed with zero pretrained weights, random initialization, RoPE embeddings,
non-autoregressive marker gathering, hierarchical high-cardinality choice,
and calibrated typed decision heads.
"""

import math
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ApexStudentConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with dynamic context scaling."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0, scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32) / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim // 2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim // 2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = torch.repeat_interleave(cos, 2, dim=-1)  # [1, 1, seq_len, head_dim]
    sin = torch.repeat_interleave(sin, 2, dim=-1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class StudentAttention(nn.Module):
    """Multi-Head Scaled Dot-Product Attention with RoPE."""
    def __init__(self, config: ApexStudentConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Native PyTorch SDPA (Scaled Dot-Product Attention with Flash/Cutlass kernel where available)
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [b, s] (1 = keep, 0 = mask) -> convert to boolean mask for SDPA
            attn_mask = attention_mask[:, None, None, :].bool()  # [b, 1, 1, s]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(b, s, self.hidden_size)
        return self.out_proj(out)


class StudentTransformerLayer(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm (norm_first=True)."""
    def __init__(self, config: ApexStudentConfig):
        super().__init__()
        self.attn = StudentAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.dropout)
        )
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN attention
        norm_x = self.ln1(x)
        attn_out = self.attn(norm_x, cos, sin, attention_mask)
        x = x + self.drop(attn_out)
        # Pre-LN MLP
        x = x + self.mlp(self.ln2(x))
        return x


class ApexStudentModel(nn.Module):
    """
    ApexStudent Neural Decision Engine.
    Trained completely from scratch with random initialization.
    """
    def __init__(self, config: Optional[ApexStudentConfig] = None):
        super().__init__()
        self.config = config or ApexStudentConfig()

        # 1. Token Embeddings & RoPE
        self.embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=self.config.pad_token_id
        )
        self.embed_ln = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(self.config.dropout)

        self.rotary = RotaryEmbedding(
            dim=self.config.hidden_size // self.config.num_attention_heads,
            max_seq_len=self.config.max_position_embeddings,
            base=self.config.rope_base,
            scaling_factor=self.config.rope_scaling_factor
        )

        # 2. Transformer Encoder Layers
        self.layers = nn.ModuleList([
            StudentTransformerLayer(self.config) for _ in range(self.config.num_hidden_layers)
        ])
        self.final_ln = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

        # 3. Question Type Conditioning: (choice=0, score=1, noul=2, verify=3)
        self.type_emb = nn.Embedding(self.config.num_qtypes, self.config.hidden_size)

        # 4. Optional Decision Head Transformer Layers
        if self.config.head_layers > 0:
            self.head_layers = nn.ModuleList([
                StudentTransformerLayer(self.config) for _ in range(self.config.head_layers)
            ])
        else:
            self.head_layers = None

        # 5. Non-Autoregressive Marker Scorer
        d = self.config.hidden_size
        self.scorer = nn.Sequential(
            nn.LayerNorm(d, eps=self.config.layer_norm_eps),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1)
        )

        # 6. Hierarchical Coarse Router for High-Cardinality Choice (up to 255 options)
        self.coarse_scorer = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 7. Action & Epistemic Abstention Head
        # Features: [pooled_cls (d), top2_0, top2_0 - top2_1, norm_entropy, k / 255.0]
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 128),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, self.config.n_act)
        )

        # 8. Evidence Span Pointer Head (Start & End Logits)
        self.span_head = nn.Linear(d, 2)

        # Calibration Buffers
        self.register_buffer("temperature", torch.tensor(self.config.default_temperature, dtype=torch.float32))

        # Initialize all parameters from scratch
        self._init_weights()

    def _init_weights(self):
        """Standard Xavier / Normal initialization for truly from-scratch learning."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
        detach_encoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single parallel forward pass evaluating typed questions over state.

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] 1 for valid tokens, 0 for pad
            marker_pos: [B, K] sequence indices of [MASK] markers for each option
            marker_mask: [B, K] boolean mask of active options (True = active, False = pad)
            qtype: [B] integer question type (0=choice, 1=score, 2=noul, 3=verify)
            detach_encoder: whether to freeze encoder gradients during head-only updates

        Returns:
            logits: [B, K] option decision logits
            act_logits: [B, n_act] action / abstention logits (0=proceed, 1=abstain)
            span_logits: [B, L, 2] start and end span token logits
        """
        b, s = input_ids.shape
        cos, sin = self.rotary(s, input_ids.device)

        # 1. Embeddings + Condition with Question Type
        h = self.embeddings(input_ids)
        h = self.embed_ln(h)
        h = self.embed_dropout(h)

        # Broadcast type embedding across sequence
        t_emb = self.type_emb(qtype)[:, None, :]  # [B, 1, d]
        h = h + t_emb

        # 2. Transformer Encoder Layers
        for layer in self.layers:
            h = layer(h, cos, sin, attention_mask)

        h = self.final_ln(h)

        if detach_encoder:
            h = h.detach()

        # 3. Decision Head Layers (if enabled)
        if self.head_layers is not None:
            for h_layer in self.head_layers:
                h = h_layer(h, cos, sin, attention_mask)

        # 4. Marker Gathering at [MASK] Positions
        # idx: [B, K, d]
        k_max = marker_pos.size(1)
        idx = marker_pos.clamp(min=0, max=max(0, h.size(1) - 1)).unsqueeze(-1).expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)  # [B, K, d]

        # 5. Hierarchical Option Scoring
        if k_max > self.config.coarse_candidate_k:
            # High-cardinality 2-stage coarse-to-fine selection
            coarse_scores = self.coarse_scorer(m).squeeze(-1)  # [B, K]
            coarse_scores = coarse_scores.masked_fill(~marker_mask, -1e4)

            # Select top candidates for deep scoring
            top_k = min(self.config.coarse_candidate_k, k_max)
            top_idx = torch.topk(coarse_scores, k=top_k, dim=-1).indices  # [B, top_k]

            # Gather top candidate vectors
            top_idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, m.size(-1))
            m_top = torch.gather(m, 1, top_idx_expanded)  # [B, top_k, d]
            fine_scores = self.scorer(m_top).squeeze(-1)  # [B, top_k]

            # Reconstruct full logits
            logits = coarse_scores.clone()
            logits.scatter_(1, top_idx, fine_scores)
        else:
            # Direct marker scoring for standard option counts
            logits = self.scorer(m).squeeze(-1).float()

        logits = logits.masked_fill(~marker_mask, -1e4)

        # 6. Action / Abstention & Calibrated Confidence Estimation
        p = torch.softmax(logits.detach(), -1)
        k_cnt = marker_mask.sum(-1).clamp(min=2).float()
        # Normalized Shannon entropy
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / torch.log(k_cnt)

        # Top-2 margins
        sorted_p, _ = p.topk(2, dim=-1)
        top1 = sorted_p[:, 0]
        top2 = sorted_p[:, 1]
        margin = top1 - top2

        feats = torch.stack([top1, margin, ent, k_cnt / float(self.config.max_options)], dim=-1)
        pooled_cls = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled_cls, feats], dim=-1))

        # 7. Evidence Span Locality
        span_logits = self.span_head(h)

        return logits, act_logits, span_logits
