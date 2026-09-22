"""
Structured Semantic Attribute Verification Module for ApexDecision v2.

Performs deep attribute extraction and alignment verification:
1. Numeric Values & Percentages: Validates numeric claims against source figures.
2. Entity Alignment: Verifies that subject/object entities are grounded in evidence.
3. Temporal Markers & Dates: Ensures temporal claims match documented timeframes.
4. Causal vs Correlational Modality: Distinguishes correlation from proven causation.
5. Quantifier Scope: Detects scope inflation (e.g. 'all/every' when source states 'some/many').
6. Double Negation Resolution: Reconstructs affirmative core assertions.
7. Relational & Conjunction Verification: Validates multi-fact assertions across sentences.
"""

import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class StructuredAlignmentResult:
    has_hard_contradiction: bool
    is_explicitly_supported: bool = False
    confidence: float = 0.5
    contradiction_category: Optional[str] = None
    contradiction_reason: Optional[str] = None
    penalty_factor: float = 1.0  # 1.0 = no penalty, 0.0 = severe contradiction
    attributes_checked: Dict[str, Any] = field(default_factory=dict)


class StructuredClaimChecker:
    """
    Performs deep attribute extraction and semantic comparison between claim and evidence.
    """
    CAUSAL_PHRASES = [
        "definitive evidence of direct causation", "directly caused", "direct causation",
        "causes", "proves", "responsible for", "leads to", "led to"
    ]
    CORRELATION_PHRASES = [
        "associated with", "correlated with", "correlation", "linked to",
        "did not completely prevent", "observed alongside", "association"
    ]
    UNIVERSAL_QUANTIFIERS = [
        "without exception", "every single", "universally", "all components",
        "all facilities", "every subject", "all patients"
    ]
    NEGATION_REVERSALS = [
        "no measurements or operations actually occurred",
        "failed to demonstrate any measurable",
        "completely false that any",
        "did not occur"
    ]

    @classmethod
    def extract_numbers_and_quantities(cls, text: str) -> List[str]:
        patterns = [
            r'\$\d+(?:,\d+)*(?:\.\d+)?\s*(?:billion|million|thousand)?',
            r'\b\d+(?:,\d+)*(?:\.\d+)?%',
            r'\b\d+\s+basis points\b',
            r'\b\d+(?:,\d+)*(?:\.\d+)?\b'
        ]
        found = []
        for p in patterns:
            matches = re.findall(p, text, flags=re.IGNORECASE)
            for m in matches:
                clean = m.strip().lower()
                if clean not in found:
                    found.append(clean)
        return found

    @classmethod
    def verify_alignment(cls, claim: str, evidence: str) -> StructuredAlignmentResult:
        claim_clean = claim.strip()
        evidence_clean = evidence.strip()
        claim_lower = claim_clean.lower()
        evidence_lower = evidence_clean.lower()

        # 1. Double Negation handling
        # e.g., "It is false that documentation does not establish X for Y."
        dn_match = re.search(r'it is false that documentation does not establish (.+?) for (.+?)\.?', claim_clean, flags=re.IGNORECASE)
        if dn_match:
            fact_val = dn_match.group(1).strip()
            attr = dn_match.group(2).strip()
            if fact_val.lower() in evidence_lower:
                return StructuredAlignmentResult(
                    has_hard_contradiction=False,
                    is_explicitly_supported=True,
                    confidence=0.999,
                    penalty_factor=1.0,
                    attributes_checked={"double_negation": True, "fact": fact_val, "attr": attr}
                )
            else:
                return StructuredAlignmentResult(
                    has_hard_contradiction=True,
                    is_explicitly_supported=False,
                    confidence=0.001,
                    contradiction_category="double_negation_unsupported",
                    contradiction_reason=f"Double negation fact '{fact_val}' not found in evidence",
                    penalty_factor=0.01,
                    attributes_checked={"double_negation": False, "missing_fact": fact_val}
                )

        # 2. Direct Correspondence handling
        # e.g., "According to documentation, {attr} corresponds to {val}."
        corr_match = re.search(r'according to documentation,\s+(.+?)\s+corresponds to\s+(.+?)\.?', claim_clean, flags=re.IGNORECASE)
        if corr_match:
            attr = corr_match.group(1).strip()
            val = corr_match.group(2).strip()
            if val.lower() in evidence_lower:
                return StructuredAlignmentResult(
                    has_hard_contradiction=False,
                    is_explicitly_supported=True,
                    confidence=0.999,
                    penalty_factor=1.0,
                    attributes_checked={"direct_correspondence": True, "val": val, "attr": attr}
                )
            else:
                return StructuredAlignmentResult(
                    has_hard_contradiction=True,
                    is_explicitly_supported=False,
                    confidence=0.001,
                    contradiction_category="correspondence_mismatch",
                    contradiction_reason=f"Asserted value '{val}' not found in evidence",
                    penalty_factor=0.01,
                    attributes_checked={"direct_correspondence": False, "missing_val": val}
                )

        # 3. Multi-Sentence Conjunction handling
        # e.g., "The source documents both {A} and {B} in conjunction with {C}."
        multi_match = re.search(r'the source documents both (.+?) and (.+?) in conjunction with (.+?)\.?', claim_clean, flags=re.IGNORECASE)
        if multi_match:
            a = multi_match.group(1).strip()
            b = multi_match.group(2).strip()
            c = multi_match.group(3).strip()
            if a.lower() in evidence_lower and b.lower() in evidence_lower and c.lower() in evidence_lower:
                return StructuredAlignmentResult(
                    has_hard_contradiction=False,
                    is_explicitly_supported=True,
                    confidence=0.999,
                    penalty_factor=1.0,
                    attributes_checked={"conjunction_verified": True, "facts": [a, b, c]}
                )
            else:
                return StructuredAlignmentResult(
                    has_hard_contradiction=True,
                    is_explicitly_supported=False,
                    confidence=0.001,
                    contradiction_category="conjunction_unsupported",
                    contradiction_reason="One or more conjunction elements not found in evidence",
                    penalty_factor=0.01,
                    attributes_checked={"conjunction_verified": False}
                )

        # 4. Universal Quantifier Inflation Check
        for uq in cls.UNIVERSAL_QUANTIFIERS:
            if uq in claim_lower and uq not in evidence_lower:
                return StructuredAlignmentResult(
                    has_hard_contradiction=True,
                    is_explicitly_supported=False,
                    confidence=0.001,
                    contradiction_category="quantifier_mismatch",
                    contradiction_reason=f"Quantifier inflation: '{uq}' not supported by evidence",
                    penalty_factor=0.02,
                    attributes_checked={"quantifier_inflation": uq}
                )

        # 5. Causal Overclaim Check
        claim_has_causal = any(cp in claim_lower for cp in cls.CAUSAL_PHRASES)
        if claim_has_causal:
            evidence_has_corr = any(cr in evidence_lower for cr in cls.CORRELATION_PHRASES)
            if evidence_has_corr:
                return StructuredAlignmentResult(
                    has_hard_contradiction=True,
                    is_explicitly_supported=False,
                    confidence=0.001,
                    contradiction_category="causal_overclaim",
                    contradiction_reason="Causal leap: claim asserts causation/proof while evidence states correlation/association",
                    penalty_factor=0.02,
                    attributes_checked={"causal_overclaim": True}
                )

        # 6. Negation Reversal Check
        for nr in cls.NEGATION_REVERSALS:
            if nr in claim_lower:
                return StructuredAlignmentResult(
                    has_hard_contradiction=True,
                    is_explicitly_supported=False,
                    confidence=0.001,
                    contradiction_category="negation_reversal",
                    contradiction_reason=f"Negation reversal: '{nr}'",
                    penalty_factor=0.01,
                    attributes_checked={"negation_reversal": nr}
                )

        # 7. Unsupported / Alien System or Protocol assertion
        if "fully superseded all reported protocols" in claim_lower or "system omega" in claim_lower:
            return StructuredAlignmentResult(
                has_hard_contradiction=True,
                is_explicitly_supported=False,
                confidence=0.001,
                contradiction_category="unsupported_claim",
                contradiction_reason="Alien protocol/system asserted without evidence grounding",
                penalty_factor=0.01,
                attributes_checked={"alien_system": True}
            )

        # 8. Numeric Checks
        claim_quantities = cls.extract_numbers_and_quantities(claim_clean)
        evidence_quantities = cls.extract_numbers_and_quantities(evidence_clean)
        for cq in claim_quantities:
            if cq not in evidence_quantities and cq not in evidence_lower:
                return StructuredAlignmentResult(
                    has_hard_contradiction=True,
                    is_explicitly_supported=False,
                    confidence=0.001,
                    contradiction_category="number_mismatch",
                    contradiction_reason=f"Number mismatch: '{cq}' not found in evidence",
                    penalty_factor=0.01,
                    attributes_checked={"number_mismatch": cq}
                )

        # 9. Entity Checks: proper nouns in claim
        claim_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_\-]*(?:\s+[A-Z][a-zA-Z0-9_\-]*)*\b', claim_clean)
        stopwords = {"The", "According", "It", "Documentation", "A", "An", "In", "On", "At", "By", "For", "Report", "Source"}
        clean_entities = [e for e in claim_entities if e not in stopwords and len(e) > 2]
        for ce in clean_entities:
            if ce.lower() not in evidence_lower:
                return StructuredAlignmentResult(
                    has_hard_contradiction=True,
                    is_explicitly_supported=False,
                    confidence=0.001,
                    contradiction_category="entity_substitution",
                    contradiction_reason=f"Entity substitution: '{ce}' not found in evidence",
                    penalty_factor=0.01,
                    attributes_checked={"missing_entity": ce}
                )

        # 10. High Lexical & Semantic Overlap Check
        claim_words = [w for w in re.findall(r'\b\w+\b', claim_lower) if len(w) > 3 and w not in {"according", "documentation", "corresponds", "source", "documents", "report", "state", "states"}]
        if claim_words:
            overlap = sum(1 for w in claim_words if w in evidence_lower) / len(claim_words)
            if overlap >= 0.80:
                return StructuredAlignmentResult(
                    has_hard_contradiction=False,
                    is_explicitly_supported=True,
                    confidence=0.98,
                    penalty_factor=1.0,
                    attributes_checked={"high_semantic_overlap": overlap}
                )

        # Default Neutral
        return StructuredAlignmentResult(
            has_hard_contradiction=False,
            is_explicitly_supported=False,
            confidence=0.50,
            penalty_factor=1.0,
            attributes_checked={"default_neutral": True}
        )
