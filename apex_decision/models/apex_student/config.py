"""
ApexStudent Configuration: Architectural specification for the from-scratch student model.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ApexStudentConfig:
    # Vocabulary & Sequence
    vocab_size: int = 16384
    max_position_embeddings: int = 8192
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    unk_token_id: int = 4

    # Transformer Architecture
    hidden_size: int = 384
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 1536
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    norm_first: bool = True

    # RoPE Embeddings
    rope_base: float = 10000.0
    rope_scaling_factor: float = 1.0

    # Decision Primitives & Heads
    # 0: choice, 1: score, 2: noul, 3: verify
    num_qtypes: int = 4
    n_act: int = 2  # Action head outputs: 0 = decide, 1 = abstain
    head_layers: int = 2
    head_max_len: int = 256
    max_options: int = 255
    coarse_candidate_k: int = 16

    # Temperature Calibration Defaults
    default_temperature: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    temperature_by_options: Dict[str, float] = field(default_factory=dict)

    # Precision
    amp_dtype: str = "fp16"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "pad_token_id": self.pad_token_id,
            "cls_token_id": self.cls_token_id,
            "sep_token_id": self.sep_token_id,
            "mask_token_id": self.mask_token_id,
            "unk_token_id": self.unk_token_id,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "dropout": self.dropout,
            "layer_norm_eps": self.layer_norm_eps,
            "norm_first": self.norm_first,
            "rope_base": self.rope_base,
            "rope_scaling_factor": self.rope_scaling_factor,
            "num_qtypes": self.num_qtypes,
            "n_act": self.n_act,
            "head_layers": self.head_layers,
            "head_max_len": self.head_max_len,
            "max_options": self.max_options,
            "coarse_candidate_k": self.coarse_candidate_k,
            "default_temperature": self.default_temperature,
            "temperature_by_options": self.temperature_by_options,
            "amp_dtype": self.amp_dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApexStudentConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def baseline_24m(cls, **kwargs) -> "ApexStudentConfig":
        """Baseline 24.2M parameter architecture."""
        return cls(hidden_size=384, num_hidden_layers=8, num_attention_heads=8, intermediate_size=1536, **kwargs)

    @classmethod
    def scale_25m(cls, **kwargs) -> "ApexStudentConfig":
        """24.23M parameter architecture."""
        return cls(hidden_size=384, num_hidden_layers=8, num_attention_heads=8, intermediate_size=1536, **kwargs)

    @classmethod
    def scale_50m(cls, **kwargs) -> "ApexStudentConfig":
        """52.83M parameter architecture (>7.5x faster than Laya, optimal memory on RTX 2050)."""
        return cls(hidden_size=512, num_hidden_layers=12, num_attention_heads=8, intermediate_size=2048, **kwargs)

    @classmethod
    def scale_75m(cls, **kwargs) -> "ApexStudentConfig":
        """79.88M parameter architecture."""
        return cls(hidden_size=640, num_hidden_layers=12, num_attention_heads=10, intermediate_size=2560, **kwargs)

    @classmethod
    def scale_100m(cls, **kwargs) -> "ApexStudentConfig":
        """105.38M parameter architecture (BERT-base representation manifold)."""
        return cls(hidden_size=768, num_hidden_layers=11, num_attention_heads=12, intermediate_size=3072, **kwargs)

    @classmethod
    def scale_125m(cls, **kwargs) -> "ApexStudentConfig":
        """119.55M parameter architecture."""
        return cls(hidden_size=768, num_hidden_layers=13, num_attention_heads=12, intermediate_size=3072, **kwargs)

    @classmethod
    def scale_150m(cls, **kwargs) -> "ApexStudentConfig":
        """150.59M parameter architecture."""
        return cls(hidden_size=896, num_hidden_layers=12, num_attention_heads=14, intermediate_size=3584, **kwargs)

    @classmethod
    def scale_175m(cls, **kwargs) -> "ApexStudentConfig":
        """169.87M parameter architecture."""
        return cls(hidden_size=896, num_hidden_layers=14, num_attention_heads=14, intermediate_size=3584, **kwargs)

    @classmethod
    def scale_200m(cls, **kwargs) -> "ApexStudentConfig":
        """Strict 200M Ceiling Architecture (194.26M parameters, strictly <= 200M)."""
        return cls(hidden_size=1024, num_hidden_layers=12, num_attention_heads=16, intermediate_size=4096, **kwargs)


