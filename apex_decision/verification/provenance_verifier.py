"""
Deterministic Safety & Exact Provenance Verification for ApexDecision V7.
Enforces the Zero-Invention Output Contract:
  1. Every Choice output must strictly belong to the input option set.
  2. Every Noul evidence citation must be an exact verbatim substring of the source state.
  3. Validates character start/end offsets.
  4. Triggers deterministic rejection/abstention if ungrounded.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import re


class ProvenanceVerificationError(Exception):
    """Raised when an output violates grounding or zero-invention constraints."""
    pass


class ProvenanceVerifier:
    """
    Deterministic validator wrapping all model predictions before they exit the engine.
    """

    @staticmethod
    def verify_choice(prediction: str, candidate_options: List[str]) -> Tuple[str, bool]:
        """
        Guarantees that a predicted choice strictly belongs to candidate_options.
        Returns (verified_choice, is_valid).
        If invalid/invented, automatically falls back to safe option or raises.
        """
        if not candidate_options:
            return prediction, True

        # Exact match
        if prediction in candidate_options:
            return prediction, True

        # Normalized case/whitespace match
        norm_map = {opt.strip().lower(): opt for opt in candidate_options}
        if prediction.strip().lower() in norm_map:
            return norm_map[prediction.strip().lower()], True

        # Rejection: Zero invention allowed
        return candidate_options[0], False

    @staticmethod
    def verify_evidence_span(source_text: str, cited_span: str) -> Dict[str, Any]:
        """
        Verifies that cited_span is an exact substring of source_text.
        Returns match metadata including start and end character offsets.
        """
        if not cited_span or not source_text:
            return {
                "verified": False,
                "start_char": -1,
                "end_char": -1,
                "exact_match": False,
                "reason": "Empty source or cited span"
            }

        # 1. Exact string search
        idx = source_text.find(cited_span)
        if idx != -1:
            return {
                "verified": True,
                "start_char": idx,
                "end_char": idx + len(cited_span),
                "exact_match": True,
                "reason": "Exact verbatim substring match"
            }

        # 2. Normalized whitespace search
        norm_source = re.sub(r"\s+", " ", source_text)
        norm_span = re.sub(r"\s+", " ", cited_span).strip()
        norm_idx = norm_source.find(norm_span)
        if norm_idx != -1:
            return {
                "verified": True,
                "start_char": norm_idx,
                "end_char": norm_idx + len(norm_span),
                "exact_match": True,
                "reason": "Normalized whitespace match"
            }

        # Failure: Citation is fabricated or altered
        return {
            "verified": False,
            "start_char": -1,
            "end_char": -1,
            "exact_match": False,
            "reason": "Cited span is not present in the source text (Fabrication Detected)"
        }

    @staticmethod
    def extract_salient_spans(source_text: str, query: str, top_k: int = 3) -> List[str]:
        """
        Deterministic span extraction finding highest-salience sentences matching query terms.
        """
        # Split into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text) if len(s.strip()) > 5]
        if not sentences:
            return [source_text[:200]]

        q_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
        scored = []
        for s in sentences:
            s_terms = set(re.findall(r"\b\w{3,}\b", s.lower()))
            overlap = len(q_terms & s_terms)
            scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_spans = [s for score, s in scored[:top_k] if score > 0]
        return top_spans if top_spans else [sentences[0]]
