"""
Hard Negative & Hard Positive Generator (v2) for ApexDecision Grounding.

Generates:
1. Hard Negatives:
   - Subtle numerical mutations (12% -> 21%)
   - Unit mismatches (kg -> mg, miles -> km)
   - Entity substitutions (Group X -> Group Y, Company A -> Company B)
   - Temporal inversions (2022 -> 2023, before -> after)
   - Quantifier scope inflation ('some/many' -> 'every/all/universally')
   - Causal inflation ('associated with' -> 'directly caused by/proves')
   - Comparative reversals (faster -> slower, higher -> lower)

2. Hard Positives (Crucial for Direct Entailment Recall):
   - High-variance syntactic paraphrases
   - Reordered clauses ('A happened because B' <-> 'Because B, A happened')
   - Multi-sentence logical synthesis (500 total, 250 treated -> 50% treated)
   - Synonym substitutions (decrease -> reduction, acquire -> purchase)
   - Numerical equivalence (1 in 4 -> 25%, double -> 100% increase)
   - Pronominal and co-reference resolution
"""

import re
import copy
import random
from typing import List, Dict, Tuple, Optional
from .dataset_schema import GroundingExample, GroundingLabel, FailureMode, DifficultyTier, Domain


class HardNegativeMinerV2:
    """
    Generates balanced hard negative and hard positive variants of grounding examples
    to train models to overcome conservative false-negative bias without increasing false positives.
    """
    SYNONYM_PAIRS = [
        ("increased", "rose"),
        ("decreased", "dropped"),
        ("acquired", "purchased"),
        ("discovered", "identified"),
        ("demonstrated", "showed"),
        ("approved", "authorized"),
        ("reduced", "lowered"),
        ("enhanced", "improved"),
        ("eliminated", "removed"),
        ("associated with", "linked to"),
        ("prior to", "before"),
        ("subsequent to", "after"),
        ("approximately", "about"),
        ("substantial", "significant"),
        ("prohibited", "banned")
    ]

    UNIT_SWAPS = {
        "mg": "kg",
        "kg": "g",
        "meters": "kilometers",
        "miles": "meters",
        "seconds": "minutes",
        "hours": "days",
        "usd": "eur",
        "dollars": "pounds",
        "percent": "basis points"
    }

    @classmethod
    def mutate_to_hard_negative(cls, example: GroundingExample, mutation_type: str) -> Optional[GroundingExample]:
        """
        Takes a SUPPORTED example and mutates the claim into a subtly NOT_SUPPORTED near-miss claim.
        """
        claim = example.claim
        mutated_claim = None
        failure_mode = FailureMode.UNSUPPORTED_CLAIM

        if mutation_type == "number_substitution":
            nums = re.findall(r'\b\d+\b', claim)
            if nums:
                n = nums[0]
                # Invert digits or add 1
                new_n = n[::-1] if (len(n) > 1 and n[::-1] != n) else str(int(n) + 5)
                mutated_claim = re.sub(rf'\b{re.escape(n)}\b', new_n, claim, count=1)
                failure_mode = FailureMode.NUMBER_MISMATCH

        elif mutation_type == "unit_substitution":
            for u1, u2 in cls.UNIT_SWAPS.items():
                if re.search(rf'\b{re.escape(u1)}\b', claim, flags=re.IGNORECASE):
                    mutated_claim = re.sub(rf'\b{re.escape(u1)}\b', u2, claim, flags=re.IGNORECASE, count=1)
                    failure_mode = FailureMode.UNIT_MISMATCH
                    break

        elif mutation_type == "entity_substitution":
            entities = re.findall(r'\b(?:Group|Cohort|Company|Model|Patient)\s+([A-Z0-9]+)\b', claim, flags=re.IGNORECASE)
            if entities:
                ent_val = entities[0]
                new_val = "Y" if ent_val.upper() == "X" else ("X" if ent_val.upper() == "Y" else ("B" if ent_val.upper() == "A" else "Z"))
                mutated_claim = re.sub(rf'\b{re.escape(ent_val)}\b', new_val, claim, count=1)
                failure_mode = FailureMode.ENTITY_SUBSTITUTION

        elif mutation_type == "causal_inflation":
            for corr in ["associated with", "linked to", "observed alongside", "correlated with"]:
                if corr in claim.lower():
                    mutated_claim = re.sub(rf'\b{re.escape(corr)}\b', "directly caused by", claim, flags=re.IGNORECASE, count=1)
                    failure_mode = FailureMode.CAUSAL_OVERCLAIM
                    break
            if not mutated_claim and "showed" in claim.lower():
                mutated_claim = re.sub(r'\bshowed\b', "proved definitively that it caused", claim, flags=re.IGNORECASE, count=1)
                failure_mode = FailureMode.CAUSAL_OVERCLAIM

        elif mutation_type == "quantifier_inflation":
            for part in ["some", "several", "a portion of", "most"]:
                if re.search(rf'\b{re.escape(part)}\b', claim, flags=re.IGNORECASE):
                    mutated_claim = re.sub(rf'\b{re.escape(part)}\b', "without exception all", claim, flags=re.IGNORECASE, count=1)
                    failure_mode = FailureMode.QUANTIFIER_MISMATCH
                    break

        elif mutation_type == "temporal_substitution":
            years = re.findall(r'\b(?:19|20)\d{2}\b', claim)
            if years:
                y = years[0]
                new_y = str(int(y) + 1)
                mutated_claim = re.sub(rf'\b{re.escape(y)}\b', new_y, claim, count=1)
                failure_mode = FailureMode.DATE_TIME_MISMATCH

        if mutated_claim and mutated_claim != claim:
            return GroundingExample(
                id=f"{example.id}_hn_{mutation_type[:4]}",
                domain=example.domain,
                source=example.source,
                claim=mutated_claim,
                label=GroundingLabel.NOT_SUPPORTED,
                failure_mode=failure_mode,
                difficulty_tier=DifficultyTier.TIER_3_ADVERSARIAL_NEAR,
                is_adversarial=True,
                evidence_spans=example.evidence_spans,
                metadata={"base_example_id": example.id, "mutation": mutation_type}
            )
        return None

    @classmethod
    def mutate_to_hard_positive(cls, example: GroundingExample, mutation_type: str) -> Optional[GroundingExample]:
        """
        Takes a SUPPORTED example and generates a syntactically challenging but strictly entailed variant.
        Directly targets the direct-entailment recall deficit.
        """
        claim = example.claim
        mutated_claim = None

        if mutation_type == "synonym_substitution":
            for w1, w2 in cls.SYNONYM_PAIRS:
                if re.search(rf'\b{re.escape(w1)}\b', claim, flags=re.IGNORECASE):
                    mutated_claim = re.sub(rf'\b{re.escape(w1)}\b', w2, claim, flags=re.IGNORECASE, count=1)
                    break
                elif re.search(rf'\b{re.escape(w2)}\b', claim, flags=re.IGNORECASE):
                    mutated_claim = re.sub(rf'\b{re.escape(w2)}\b', w1, claim, flags=re.IGNORECASE, count=1)
                    break

        elif mutation_type == "clause_reorder":
            # If claim has "after X, Y happened" -> "Y happened after X"
            m = re.match(r'^(?:After|Following|In)\s+([^,]+),\s*(.+)$', claim, flags=re.IGNORECASE)
            if m:
                pre, post = m.group(1), m.group(2)
                post_clean = post[0].upper() + post[1:] if len(post) > 1 else post
                mutated_claim = f"{post_clean.rstrip('.')} after {pre.lower()}."

        elif mutation_type == "passive_voice_shift":
            # Simple passive/active inversion
            # e.g., "Company A acquired Company B" -> "Company B was acquired by Company A"
            m = re.match(r'^([A-Z][a-zA-Z0-9\s]+?)\s+(acquired|developed|purchased|tested|implemented)\s+([A-Z][a-zA-Z0-9\s]+?)(?:\s+(in\s+\d{4}))?\b', claim)
            if m:
                subj, verb, obj, time_p = m.group(1), m.group(2), m.group(3), m.group(4)
                time_str = f" {time_p}" if time_p else ""
                mutated_claim = f"{obj} was {verb} by {subj}{time_str}."

        elif mutation_type == "numerical_equivalence":
            # e.g., "reduced by 50%" <-> "cut in half"
            if "reduced by 50%" in claim.lower():
                mutated_claim = re.sub(r'reduced by 50%', 'cut in half', claim, flags=re.IGNORECASE)
            elif "25%" in claim.lower():
                mutated_claim = re.sub(r'25%', 'one in four', claim, flags=re.IGNORECASE)
            elif "doubled" in claim.lower():
                mutated_claim = re.sub(r'doubled', 'increased twofold', claim, flags=re.IGNORECASE)

        if mutated_claim and mutated_claim != claim:
            return GroundingExample(
                id=f"{example.id}_hp_{mutation_type[:4]}",
                domain=example.domain,
                source=example.source,
                claim=mutated_claim,
                label=GroundingLabel.SUPPORTED,
                failure_mode=FailureMode.DIRECT_ENTAILMENT,
                difficulty_tier=DifficultyTier.TIER_2_PARAPHRASE,
                is_adversarial=False,
                evidence_spans=example.evidence_spans,
                metadata={"base_example_id": example.id, "mutation": mutation_type}
            )
        return None
