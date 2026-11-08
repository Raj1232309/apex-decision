"""
Configuration for Apex-748M Classifier and Decision Engine.
Production-audited specification matching docs/APEX_750M_SPECIFICATION.md v3.0.0.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Apex748MConfig:
    # Model Dimensions
    vocab_size: int = 16384
    hidden_size: int = 1536
    intermediate_size: int = 4096  # SwiGLU FFN dimension
    num_attention_heads: int = 24
    head_dim: int = 64  # 1536 / 24 = 64

    # Layer Allocation
    num_hidden_layers: int = 21   # Tier 1 (Local Bidirectional Encoder)
    num_memory_layers: int = 4    # Tier 2 (Global Memory Cross-Attention)

    # Sequence & Chunking
    max_chunk_length: int = 512
    rope_base: float = 10000.0
    memory_latents_per_chunk: int = 4
    max_chunks: int = 1000        # Supports up to 512,000 tokens (1000 * 512)

    # Decision Heads Configuration
    num_score_levels: int = 11    # Ordinal tiers: 0 to 10 (requires 10 cutpoint thresholds)
    max_choice_candidates: int = 255
    dropout: float = 0.05
    layer_norm_eps: float = 1e-5

    # Initialization & Regularization
    init_std: float = 0.02
    initial_tau_choice: float = 39.1918  # sqrt(1536) ≈ 39.1918
    abstention_threshold: float = 0.65   # theta_abstain
    gradient_checkpointing: bool = True
