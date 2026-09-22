"""
Claim Decomposition Module for ApexDecision v2.

Decomposes compound, multi-predicate, or conditioned claims into atomic propositions.
Extracts entities, numbers, percentages, temporal markers, and negation state per proposition.
Runs 100% locally with deterministic rule-based semantic parsing.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AtomicProposition:
    text: str
    proposition_id: str
    numbers: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    is_negated: bool = False
    has_causal_assertion: bool = False
    modality: Optional[str] = None  # e.g., 'proven', 'observed', 'suggested', 'correlated'


class ClaimDecomposer:
    """
    Decomposes claims into atomic sub-claims so that each factual assertion
    can be independently validated against evidence spans.
    """
    CONJUNCTIONS = [
        r'\band\b',
        r'\bwhile\b',
        r'\bwhereas\b',
        r'\bbut\b',
        r'\bhowever\b',
        r';',
        r'\bas well as\b',
        r'\bin addition to\b',
    ]

    CAUSAL_MARKERS = [
        "causes", "caused", "causing", "proves", "proven", "leads to", "led to",
        "results in", "resulted in", "direct result of", "responsible for"
    ]

    CORRELATION_MARKERS = [
        "associated with", "correlated with", "linked to", "co-occurred", "observed alongside"
    ]

    UNITS = [
        "%", "percent", "percentage", "mg", "g", "kg", "ml", "l", "meters", "km",
        "miles", "ms", "seconds", "minutes", "hours", "days", "weeks", "months", "years",
        "usd", "dollars", "$", "eur", "gbp", "bps", "basis points", "patients", "participants"
    ]

    @classmethod
    def extract_numbers_and_units(cls, text: str) -> tuple[List[str], List[str]]:
        numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?%?\b', text)
        clean_nums = [n.replace(",", "") for n in numbers]
        
        found_units = []
        lower = text.lower()
        for u in cls.UNITS:
            if re.search(rf'\b{re.escape(u)}\b', lower):
                found_units.append(u)
        return clean_nums, found_units

    @classmethod
    def extract_dates(cls, text: str) -> List[str]:
        patterns = [
            r'\b(?:19|20)\d{2}\b',  # Years (e.g. 2022)
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\bq[1-4]\b',           # Quarters (e.g. Q3)
            r'\bh[1-2]\b'            # Halves (e.g. H1)
        ]
        dates = []
        for p in patterns:
            dates.extend(re.findall(p, text.lower()))
        return list(set(dates))

    @classmethod
    def extract_entities(cls, text: str) -> List[str]:
        # Extract capitalized phrases or alphanumeric identifiers (e.g., 'Model X', 'Company B', 'Group 1')
        entities = re.findall(r'\b[A-Z][a-zA-Z0-9_\-]*(?:\s+[A-Z][a-zA-Z0-9_\-]*)*\b', text)
        common_words = {"The", "A", "An", "In", "On", "At", "By", "For", "With", "After", "Before", "During"}
        return [e for e in entities if e not in common_words and len(e) > 1]

    @classmethod
    def decompose(cls, claim: str) -> List[AtomicProposition]:
        claim = claim.strip()
        if not claim:
            return []

        # Split complex sentences by compound clauses
        parts = [claim]
        for conj in cls.CONJUNCTIONS:
            new_parts = []
            for p in parts:
                splits = re.split(conj, p, flags=re.IGNORECASE)
                for s in splits:
                    s_clean = s.strip()
                    if len(s_clean.split()) >= 3:  # Only treat as distinct proposition if enough words
                        new_parts.append(s_clean)
                    elif new_parts:
                        # Append short dangling fragment to previous proposition
                        new_parts[-1] += f" {s_clean}"
                    else:
                        new_parts.append(s_clean)
            parts = new_parts

        propositions = []
        for idx, part in enumerate(parts):
            p_clean = part.strip()
            if not p_clean:
                continue

            nums, units = cls.extract_numbers_and_units(p_clean)
            dates = cls.extract_dates(p_clean)
            entities = cls.extract_entities(p_clean)

            # Negation check
            is_negated = bool(re.search(r'\b(?:not|never|no|none|neither|nor|without|failed to)\b', p_clean.lower()))

            # Causal vs correlation check
            is_causal = any(m in p_clean.lower() for m in cls.CAUSAL_MARKERS)
            is_corr = any(m in p_clean.lower() for m in cls.CORRELATION_MARKERS)
            modality = "causal" if is_causal else ("correlation" if is_corr else "assertive")

            propositions.append(AtomicProposition(
                text=p_clean,
                proposition_id=f"P{idx + 1}",
                numbers=nums,
                units=units,
                dates=dates,
                entities=entities,
                is_negated=is_negated,
                has_causal_assertion=is_causal,
                modality=modality
            ))

        return propositions if propositions else [
            AtomicProposition(
                text=claim,
                proposition_id="P1",
                numbers=cls.extract_numbers_and_units(claim)[0],
                units=cls.extract_numbers_and_units(claim)[1],
                dates=cls.extract_dates(claim),
                entities=cls.extract_entities(claim)
            )
        ]
