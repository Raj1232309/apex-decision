"""
Apex-748M Model: Standalone Non-Autoregressive Zero-Token Neural Decision Engine.
Matches docs/APEX_750M_SPECIFICATION.md v3.0.0 exactly (746,813,455 Parameters).
"""

import math
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from .config import Apex748MConfig
except (ImportError, ValueError):
    try:
        from apex_classifier_750m.config import Apex748MConfig
    except (ImportError, ValueError):
        from config import Apex748MConfig


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (2 * d params per transformer layer)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.pow(2), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with base theta = 10,000."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, num_heads, seq_len, head_dim]
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


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (Self-Attn or Cross-Attn) without biases (4 * d^2 params)."""
    def __init__(self, config: Apex748MConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, sq, _ = query.shape
        key = query if key is None else key
        value = query if value is None else value
        sk = key.shape[1]

        q = self.q_proj(query).view(b, sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, sk, self.num_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].bool()

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(b, sq, self.hidden_size)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network matching parameter math: 3 * d * d_ff (18,874,368 params)."""
    def __init__(self, config: Apex748MConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerLayer(nn.Module):
    """Full Transformer Layer (MHA + SwiGLU + 2 RMSNorms = 28,314,624 params)."""
    def __init__(self, config: Apex748MConfig):
        super().__init__()
        self.attn_ln = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ffn_ln = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN Self or Cross-Attention
        normed = self.attn_ln(x)
        if context is None:
            attn_out = self.attn(normed, cos=cos, sin=sin, mask=mask)
        else:
            attn_out = self.attn(normed, key=context, value=context, mask=mask)
        x = x + attn_out

        # Pre-LN SwiGLU FFN
        x = x + self.ffn(self.ffn_ln(x))
        return x


class NoulHead(nn.Module):
    """Proposition entailment verification head (1,184,258 parameters)."""
    def __init__(self, config: Apex748MConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, 768, bias=True)
        self.fc2 = nn.Linear(768, 1, bias=True)
        self.alpha_noul = nn.Parameter(torch.tensor(0.5413, dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.alpha_noul) + 1e-4

    def forward(self, h_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.ln(h_out)
        proj = F.gelu(self.fc1(normed))
        z = self.fc2(proj)
        p = torch.sigmoid(z / self.temperature)
        return z, p


class ScoreHead(nn.Module):
    """Ordinal Score Head via Monotonic Cumulative Link (790,539 parameters)."""
    def __init__(self, config: Apex748MConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, 512, bias=True)
        self.fc2 = nn.Linear(512, 1, bias=True)

        num_thresholds = config.num_score_levels - 1  # 10 cutpoint thresholds
        self.b0 = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.log_deltas = nn.Parameter(torch.full((num_thresholds - 1,), -0.5, dtype=torch.float32))

    def get_thresholds(self) -> torch.Tensor:
        deltas = torch.exp(self.log_deltas)
        thresholds = [self.b0]
        curr = self.b0
        for delta in deltas:
            curr = curr + delta
            thresholds.append(curr)
        return torch.stack(thresholds)

    def forward(self, h_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.ln(h_out)
        u = self.fc2(F.gelu(self.fc1(normed)))
        thresholds = self.get_thresholds()

        logits = thresholds.unsqueeze(0) - u
        C = torch.sigmoid(logits)

        c0 = C[:, 0:1]
        c_mid = C[:, 1:] - C[:, :-1]
        c_last = 1.0 - C[:, -1:]
        pmf = torch.cat([c0, c_mid, c_last], dim=-1)
        return u, pmf


class ChoiceHead(nn.Module):
    """Unified High-K Choice Head with candidate encoder W_cand (3,073 head params)."""
    def __init__(self, config: Apex748MConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.tau_choice = nn.Parameter(torch.tensor(config.initial_tau_choice, dtype=torch.float32))

    def forward(self, h_out: torch.Tensor, candidate_reprs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.ln(h_out).unsqueeze(1)  # [batch, 1, hidden_size]
        scores = torch.sum(normed * candidate_reprs, dim=-1) / self.tau_choice
        probs = F.softmax(scores, dim=-1)
        return scores, probs


class AbstentionHead(nn.Module):
    """Epistemic Abstention Head (1,537 parameters)."""
    def __init__(self, config: Apex748MConfig):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size, 1, bias=True)

    def forward(self, h_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logit = self.fc(h_out)
        prob = torch.sigmoid(logit)
        return logit, prob


class Apex748MClassifier(nn.Module):
    """
    Apex-748M Non-Autoregressive Decision Engine.
    Total Audited Parameters: 746,813,455.
    """
    def __init__(self, config: Optional[Apex748MConfig] = None):
        super().__init__()
        self.config = config or Apex748MConfig()

        # 1. Token Embeddings: 16,384 * 1536 = 25,165,824 params
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        # 2. Rotary Positional Embeddings (theta = 10,000)
        self.rotary = RotaryEmbedding(
            dim=self.config.head_dim,
            max_seq_len=self.config.max_chunk_length,
            base=self.config.rope_base
        )

        # 3. Tier 1: Local Bidirectional Encoder (21 layers * 28,314,624 = 594,607,104 params)
        self.tier1_layers = nn.ModuleList([
            TransformerLayer(self.config) for _ in range(self.config.num_hidden_layers)
        ])

        # 4. Memory Latents & Projections (9,443,328 params)
        # 4 learned latents (4 * 1536 = 6,144) + Cross-Attention Projections (4 * 1536^2 = 9,437,184)
        self.memory_tokens = nn.Parameter(torch.randn(self.config.memory_latents_per_chunk, self.config.hidden_size) * 0.02)
        self.memory_attn = MultiHeadAttention(self.config)

        # 5. Tier 2: Global Memory Cross-Attention (4 layers * 28,314,624 = 113,258,496 params)
        self.tier2_layers = nn.ModuleList([
            TransformerLayer(self.config) for _ in range(self.config.num_memory_layers)
        ])

        # 6. Candidate Projection for Choice Head: W_cand in R^(1536x1536) = 2,359,296 params
        self.w_cand = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

        # 7. Decision Heads (Shared MLPs + LayerNorms = 1,979,407 params)
        self.noul_head = NoulHead(self.config)
        self.score_head = ScoreHead(self.config)
        self.choice_head = ChoiceHead(self.config)
        self.abstain_head = AbstentionHead(self.config)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.normal_(param, mean=0.0, std=self.config.init_std)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode_chunk(self, chunk_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, seq_len = chunk_ids.shape
        x = self.embeddings(chunk_ids)
        cos, sin = self.rotary(seq_len, x.device)

        for layer in self.tier1_layers:
            if self.config.gradient_checkpointing and self.training:
                def _layer_fwd(h, c, s, m):
                    return layer(h, cos=c, sin=s, mask=m)
                x = checkpoint.checkpoint(_layer_fwd, x, cos, sin, mask, use_reentrant=False)
            else:
                x = layer(x, cos=cos, sin=sin, mask=mask)

        mem_queries = self.memory_tokens.unsqueeze(0).expand(b, -1, -1)
        latents = self.memory_attn(query=mem_queries, key=x, value=x, mask=mask)
        return latents

    def encode_candidates(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        b, k, clen = candidate_ids.shape
        flat = candidate_ids.view(b * k, clen)
        emb = self.embeddings(flat)
        mean_pooled = torch.mean(emb, dim=1)
        proj = self.w_cand(mean_pooled)
        return proj.view(b, k, self.config.hidden_size)

    def forward_context(self, memory_matrix: torch.Tensor) -> torch.Tensor:
        b = memory_matrix.shape[0]
        # Query uses mean-pooled initial memory token representation
        q = self.memory_tokens[0:1].unsqueeze(0).expand(b, -1, -1)

        for layer in self.tier2_layers:
            if self.config.gradient_checkpointing and self.training:
                def _layer_fwd_mem(h, ctx):
                    return layer(h, context=ctx)
                q = checkpoint.checkpoint(_layer_fwd_mem, q, memory_matrix, use_reentrant=False)
            else:
                q = layer(q, context=memory_matrix)

        h_out = q.squeeze(1)
        return h_out

    def predict_all(
        self,
        memory_matrix: torch.Tensor,
        candidate_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        h_out = self.forward_context(memory_matrix)

        noul_z, noul_p = self.noul_head(h_out)
        score_u, score_pmf = self.score_head(h_out)
        abstain_logit, abstain_prob = self.abstain_head(h_out)

        results = {
            "h_out": h_out,
            "noul_logit": noul_z,
            "noul_prob": noul_p,
            "score_utility": score_u,
            "score_pmf": score_pmf,
            "abstain_prob": abstain_prob,
        }

        if candidate_ids is not None:
            cand_reprs = self.encode_candidates(candidate_ids)
            choice_scores, choice_probs = self.choice_head(h_out, cand_reprs)
            results["choice_scores"] = choice_scores
            results["choice_probs"] = choice_probs

        return results

    def param_count(self) -> int:
        """Total parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        End-to-end forward pass.
        input_ids: [batch, num_chunks, chunk_len] or [num_chunks, chunk_len]
        candidate_tokens: [batch, num_cands, cand_len]
        """
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(0)

        b, num_chunks, chunk_len = input_ids.shape
        flat_chunks = input_ids.view(b * num_chunks, chunk_len)
        flat_latents = self.encode_chunk(flat_chunks, mask=mask)
        mem_matrix = flat_latents.view(b, num_chunks * self.config.memory_latents_per_chunk, self.config.hidden_size)

        preds = self.predict_all(mem_matrix, candidate_ids=candidate_tokens)
        out = {
            "h_out": preds["h_out"],
            "noul_logit": preds["noul_logit"],
            "noul_prob": preds["noul_prob"],
            "score_utility": preds["score_utility"],
            "score_probs": preds["score_pmf"],
            "abstain_prob": preds["abstain_prob"],
        }
        if "choice_scores" in preds:
            out["choice_logits"] = preds["choice_scores"]
            out["choice_probs"] = preds["choice_probs"]
        return out
