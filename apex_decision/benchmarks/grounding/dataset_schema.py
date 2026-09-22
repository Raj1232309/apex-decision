"""
Dataset schema and core data structures for the ApexDecision Grounding Benchmark.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GroundingLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class Domain(str, Enum):
    SCIENCE = "science"
    MEDICINE = "medicine"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    LAW_POLICY = "law_policy"
    BUSINESS = "business"
    GENERAL_KNOWLEDGE = "general_knowledge"
    SECURITY = "security"


class FailureMode(str, Enum):
    DIRECT_ENTAILMENT = "direct_entailment"
    DIRECT_CONTRADICTION = "direct_contradiction"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    NEGATION_REVERSAL = "negation_reversal"
    DOUBLE_NEGATION = "double_negation"
    ENTITY_SUBSTITUTION = "entity_substitution"
    NUMBER_MISMATCH = "number_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    DATE_TIME_MISMATCH = "date_time_mismatch"
    POPULATION_MISMATCH = "population_mismatch"
    SCOPE_MISMATCH = "scope_mismatch"
    QUANTIFIER_MISMATCH = "quantifier_mismatch"
    CAUSAL_OVERCLAIM = "causal_overclaim"
    CORRELATION_VS_CAUSATION = "correlation_vs_causation"
    COMPARATIVE_CLAIM_MISMATCH = "comparative_claim_mismatch"
    TEMPORAL_ORDERING_ERROR = "temporal_ordering_error"
    ATTRIBUTION_MISMATCH = "attribution_mismatch"
    PARTIAL_SUPPORT_CLAIM = "partial_support_claim"
    MULTI_SENTENCE_REASONING = "multi_sentence_reasoning"
    LEXICAL_OVERLAP_UNSUPPORTED = "lexical_overlap_unsupported"


class DifficultyTier(int, Enum):
    TIER_1_DIRECT = 1          # Direct lexical/semantic support
    TIER_2_PARAPHRASE = 2      # Paraphrased syntax / synonym support
    TIER_3_ADVERSARIAL_NEAR = 3 # Adversarial near-miss (high lexical overlap, critical subtle flaw)
    TIER_4_COMPLEX_LOGIC = 4   # Complex multi-step reasoning, scope, or causal nuance


class GroundingExample(BaseModel):
    id: str
    domain: Domain
    source: str
    claim: str
    label: GroundingLabel
    failure_mode: FailureMode
    difficulty_tier: DifficultyTier
    is_adversarial: bool
    evidence_spans: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroundingPrediction(BaseModel):
    example_id: str
    decision: GroundingLabel
    probability: float
    evidence_spans: List[str] = Field(default_factory=list)
    reason: str
    latency_us: int
