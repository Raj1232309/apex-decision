"""
Numeric & Directional Polarity Guard for ApexDecision V7.
Specialized deterministic validator addressing numerical relationships,
percentages, comparative inequalities, and algebraic signs.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Set


class NumericPolarityGuard:
    """
    Validates directional claims involving numbers, percentages, and inequalities.
    Prevents false-positive entailments caused by sign inversions (e.g. -2.30% vs -1.86%).
    """

    NUM_REGEX = re.compile(r"([-+]?\$?\d+(?:\.\d+)?%?)")

    @classmethod
    def extract_numbers_with_context(cls, text: str) -> List[Dict[str, Any]]:
        """Extracts numerical tokens with polarity context (increase, decrease, reduction, negative sign)."""
        text_lower = text.lower()
        matches = []
        for m in cls.NUM_REGEX.finditer(text_lower):
            raw_str = m.group(1)
            clean_str = raw_str.replace("$", "").replace("%", "").replace("+", "")
            try:
                val = float(clean_str)
            except ValueError:
                continue

            # Determine sign/direction from surroundings
            start = max(0, m.start() - 25)
            end = min(len(text_lower), m.end() + 25)
            window = text_lower[start:end]

            is_negative = raw_str.startswith("-") or any(
                w in window for w in ["reduction", "decrease", "drop", "fell", "loss", "lower", "declined"]
            )
            is_positive = raw_str.startswith("+") or any(
                w in window for w in ["increase", "gain", "rise", "rose", "higher", "grew"]
            )

            matches.append({
                "raw": raw_str,
                "value": val,
                "effective_sign": -1.0 if is_negative else (1.0 if is_positive else 0.0),
                "is_percent": "%" in raw_str,
                "is_currency": "$" in raw_str,
                "window": window
            })
        return matches

    @classmethod
    def check_numeric_consistency(cls, source_text: str, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Verifies numerical agreement between source and claim.
        Returns (is_consistent, conflict_reason).
        """
        source_nums = cls.extract_numbers_with_context(source_text)
        claim_nums = cls.extract_numbers_with_context(claim_text)

        if not claim_nums:
            return True, None  # No numbers in claim to verify

        if not source_nums and claim_nums:
            return False, "Claim asserts numbers that are entirely absent from source text"

        source_vals = {s["value"] for s in source_nums}
        claim_vals = {c["value"] for c in claim_nums}

        # Check for direct numerical conflict
        for c in claim_nums:
            # Look for exact or sign-adjusted match in source
            matched_sources = [s for s in source_nums if abs(s["value"] - c["value"]) < 1e-4]
            if not matched_sources:
                return False, f"Number {c['raw']} in claim has no corresponding value in source"

            # Check directional polarity
            for s in matched_sources:
                if s["effective_sign"] != 0 and c["effective_sign"] != 0:
                    if s["effective_sign"] != c["effective_sign"]:
                        return False, (
                            f"Polarity inversion: source indicates direction {s['effective_sign']} "
                            f"for {s['raw']} but claim asserts direction {c['effective_sign']} for {c['raw']}"
                        )

        return True, None

    @classmethod
    def check_numeric_contradiction(cls, source_text: str, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if a claim directly contradicts numerical facts in source text
        via numerical substitution, sign inversion, or conflicting amounts for the same clause.
        """
        source_nums = cls.extract_numbers_with_context(source_text)
        claim_nums = cls.extract_numbers_with_context(claim_text)
        if not claim_nums:
            return False, None

        s_lower = source_text.lower()
        s_words = set(re.findall(r"\b\w+\b", s_lower))

        for c in claim_nums:
            c_words = set(re.findall(r"\b\w+\b", c["window"])) - {
                "is", "was", "the", "a", "an", "and", "for", "of", "in", "at", "to", "by", "this"
            }
            overlap = c_words & s_words
            # If the context is actively discussed in source (e.g. 'notice', 'days', 'written')
            if len(overlap) >= 3:
                matched = [s for s in source_nums if abs(s["value"] - c["value"]) < 1e-4]
                if not matched:
                    return True, f"Numerical substitution contradiction: claim asserts {c['raw']} in context of {list(overlap)[:3]} contrary to source"

            # Check polarity inversion
            matched_sources = [s for s in source_nums if abs(s["value"] - c["value"]) < 1e-4]
            for s in matched_sources:
                if s["effective_sign"] != 0 and c["effective_sign"] != 0 and s["effective_sign"] != c["effective_sign"]:
                    return True, f"Directional polarity contradiction: {s['effective_sign']} vs {c['effective_sign']}"

        return False, None

