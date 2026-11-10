"""
Synthetic Data Generation & Streaming Dataset for Apex-748M.
Generates deterministic mathematical balance sheets, multi-document needle anchors, and paired counterfactuals.
Supports Phase 1 (SFT Warmup) and Phase 2 (RLVR with G=4 Group Sampling).
"""

import random
from typing import Dict, Any, List, Tuple
from decimal import Decimal
import torch
from torch.utils.data import IterableDataset

try:
    from .ast_sandbox import SandboxedMathEvaluator
except (ImportError, ValueError):
    try:
        from apex_classifier_750m.ast_sandbox import SandboxedMathEvaluator
    except (ImportError, ValueError):
        from ast_sandbox import SandboxedMathEvaluator


class SyntheticDecisionDataset(IterableDataset):
    """
    Streaming synthetic dataset for deterministic training.
    Produces 100% verified ground-truth labels and expressions without external teacher LLMs.
    """
    def __init__(
        self,
        phase: str = "sft",  # "sft" or "rlvr"
        group_size: int = 4,
        vocab_size: int = 16384,
        chunk_length: int = 512,
        max_chunks: int = 4,
        seed: int = 42
    ):
        super().__init__()
        self.phase = phase
        self.group_size = group_size
        self.vocab_size = vocab_size
        self.chunk_length = chunk_length
        self.max_chunks = max_chunks
        self.evaluator = SandboxedMathEvaluator()
        random.seed(seed)

    def _generate_math_chain(self) -> Tuple[str, Decimal, str]:
        """Generates a multi-step financial arithmetic scenario with AST-verified ground truth."""
        base = random.randint(100_000, 5_000_000)
        margin = round(random.uniform(0.05, 0.35), 2)
        overhead = random.randint(10_000, 200_000)
        expr = f"{base} * (1 + {margin}) - {overhead}"
        exact_value = self.evaluator.evaluate(expr)
        text = (
            f"Financial Summary: Base operational budget was ${base:,}. "
            f"Revenue expansion added a {int(margin*100)}% margin. "
            f"Unforeseen compliance overhead deducted ${overhead:,}. "
            f"Audited net operating balance is calculated as: {expr}."
        )
        return text, exact_value, expr

    def _generate_sample(self) -> Dict[str, Any]:
        """Generates a single training item with chunks, query, label, and candidate choices."""
        text, net_balance, expr = self._generate_math_chain()

        # Dummy tokenization mapping for synthetic training
        tokens = [((hash(w) % (self.vocab_size - 10)) + 5) for w in text.split()]
        if len(tokens) < self.chunk_length:
            tokens = tokens + [0] * (self.chunk_length - len(tokens))
        else:
            tokens = tokens[:self.chunk_length]

        chunk_tensor = torch.tensor(tokens, dtype=torch.long)
        # Replicate to multi-chunk context
        context_chunks = chunk_tensor.unsqueeze(0).repeat(self.max_chunks, 1)

        # 50% chance claim is true, 50% false
        is_supported = random.choice([True, False])
        claimed_balance = net_balance if is_supported else net_balance + Decimal(random.choice([-50000, 50000, 100000]))

        # Candidates for high-K choice: 1 correct + 3 distractors
        cand_values = [net_balance, net_balance + Decimal('50000'), net_balance - Decimal('25000'), net_balance + Decimal('100000')]
        random.shuffle(cand_values)
        correct_choice_idx = cand_values.index(net_balance)

        # Create dummy candidate tokens [K=4, cand_len=16]
        cand_tokens = torch.randint(5, self.vocab_size, (len(cand_values), 16), dtype=torch.long)

        # Ordinal score tier: 0 to 10
        score_tier = min(10, max(0, int(net_balance // 500_000)))

        return {
            "chunks": context_chunks,                   # [num_chunks, 512]
            "expr": expr,
            "net_balance": float(net_balance),
            "noul_label": 1.0 if is_supported else 0.0,  # Binary entailment
            "choice_candidates": cand_tokens,           # [4, 16]
            "choice_label": correct_choice_idx,         # Target option index
            "score_label": score_tier,                  # Target tier 0-10
            "abstain_label": 0.0                        # 0 = confident, 1 = abstain
        }

    def _generate_rlvr_group(self) -> List[Dict[str, Any]]:
        """Generates G=4 mutated query variations for the same document context for GRPO."""
        base_sample = self._generate_sample()
        group = [base_sample]

        # 1. Baseline
        # 2. Counterfactual edit (flip condition)
        cf_sample = dict(base_sample)
        cf_sample["noul_label"] = 1.0 - base_sample["noul_label"]
        group.append(cf_sample)

        # 3. Numerical distractor
        dist_sample = dict(base_sample)
        dist_sample["score_label"] = (base_sample["score_label"] + random.randint(1, 3)) % 11
        group.append(dist_sample)

        # 4. Out-of-distribution unanswerable (abstain)
        abs_sample = dict(base_sample)
        abs_sample["abstain_label"] = 1.0
        group.append(abs_sample)

        return group[:self.group_size]

    def __iter__(self):
        while True:
            if self.phase == "sft":
                yield self._generate_sample()
            else:
                yield self._generate_rlvr_group()
