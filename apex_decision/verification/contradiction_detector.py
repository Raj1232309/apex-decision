"""
Contradiction & Negation Detector for ApexDecision V7.
Identifies logical incompatibilities, scope extrapolations, negation reversals,
and contractual amendment supersessions.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Set


class ContradictionDetector:
    """
    Symbolic & structural contradiction detector.
    Protects against subtle semantic conflicts and temporal supersessions.
    """

    NEGATION_TERMS = {
        "not", "never", "no", "neither", "nor", "none", "zero",
        "void", "refused", "denied", "prohibited", "banned", "without"
    }

    SUPERSEDED_PATTERNS = [
        re.compile(r"superseded\s+by\s+(?:amendment|section|clause|paragraph)\s*([a-zA-Z0-9\.]+)", re.IGNORECASE),
        re.compile(r"superseded\s+(?:prior|clause|section)\s*([a-zA-Z0-9\.]+)", re.IGNORECASE),
        re.compile(r"(?:section|clause|paragraph)\s*([0-9\.]+).*?(?:superseded|null and void|void)", re.IGNORECASE),
        re.compile(r"amendment\s+(?:no\.\s*)?(\d+).+?superseded", re.IGNORECASE),
        re.compile(r"replaces\s+section\s*([a-zA-Z0-9\.]+)", re.IGNORECASE)
    ]

    CORE_NEGATORS = {"not", "never", "no", "neither", "nor", "none", "cannot", "couldnt", "didnt", "wasnt", "isnt"}

    DOUBLE_NEG_PATTERNS = [
        re.compile(r"\b(?:not|never)\s+(?:inaccurate|incorrect|untrue|unlikely|impossible|uncommon|unusual|without)\b", re.IGNORECASE),
        re.compile(r"\bno\s+lack\s+of\b", re.IGNORECASE)
    ]

    @staticmethod
    def _stem(w: str) -> str:
        w = w.lower()
        for suffix in ["ing", "tion", "ed", "es", "s"]:
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                return w[:-len(suffix)]
        return w

    @classmethod
    def check_negation_conflict(cls, source_text: str, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if a claim directly inverts a negation present in the salient evidence sentence.
        Correctly accounts for double negations (e.g. 'not inaccurate' -> affirmed).
        """
        c_words = set(re.findall(r"\b\w+\b", claim_text.lower()))
        c_has_neg = bool(c_words & cls.CORE_NEGATORS)
        if any(p.search(claim_text) for p in cls.DOUBLE_NEG_PATTERNS):
            c_has_neg = False

        # Segment source into candidate sentences to isolate the salient claim-relevant proposition
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text) if s.strip()]
        if not sentences:
            sentences = [source_text]

        # Find the sentence with highest non-stopword lexical overlap with the claim
        stop_words = {"is", "was", "the", "a", "an", "and", "in", "of", "to", "for", "with", "that", "this", "by", "at", "did", "does"}
        c_content = c_words - stop_words - cls.CORE_NEGATORS
        c_stems = {cls._stem(w) for w in c_content}

        best_sent = sentences[0]
        best_overlap = 0
        best_matched_stems = set()
        for sent in sentences:
            s_content = set(re.findall(r"\b\w+\b", sent.lower())) - stop_words - cls.CORE_NEGATORS
            s_stems = {cls._stem(w) for w in s_content}
            overlap = s_stems & c_stems
            if len(overlap) > best_overlap:
                best_overlap = len(overlap)
                best_sent = sent
                best_matched_stems = overlap

        # Evaluate polarity conflict if there is strong predicate overlap (>= 3 stemmed content words)
        if best_overlap >= 3:
            s_words = set(re.findall(r"\b\w+\b", best_sent.lower()))
            s_has_neg = bool(s_words & cls.CORE_NEGATORS)
            if any(p.search(best_sent) for p in cls.DOUBLE_NEG_PATTERNS):
                s_has_neg = False

            # Sentences establishing constraints / thresholds (e.g. 'must not fall below X', 'shall not exceed X')
            # affirm the existence of a threshold. If claim specifies/asserts a threshold, this is not a conflict.
            constraint_neg_patterns = [
                r"not\s+(?:fall\s+below|drop\s+below|go\s+below|dip\s+below|exceed|breach|surpass|decrease\s+below)",
                r"never\s+(?:fall\s+below|drop\s+below|go\s+below|exceed|breach|surpass)",
                r"no\s+less\s+than",
                r"no\s+more\s+than"
            ]
            if any(re.search(p, best_sent, re.IGNORECASE) for p in constraint_neg_patterns):
                threshold_claim_words = {"threshold", "limit", "minimum", "maximum", "cap", "floor", "mandate", "standard", "guideline", "specify", "specifies", "specified"}
                if c_words & threshold_claim_words:
                    s_has_neg = False

            # Claims asserting absence of threshold or exemption contradict a threshold mandate
            if any(re.search(p, best_sent, re.IGNORECASE) for p in constraint_neg_patterns) or "mandate" in best_sent.lower():
                if any(w in claim_text.lower() for w in ["is exempt", "no threshold", "exempt and has no threshold"]):
                    return True, "Mandated threshold contradicted by exemption claim"

            # Direct contradiction on zero downtime / outage
            if re.search(r"\b(?:zero|no)\s+(?:downtime|outage|interruption)\b", best_sent, re.IGNORECASE):
                if re.search(r"\b(?:multiple\s+hours\s+of\s+|significant\s+|extended\s+)?(?:downtime|outage|interruption)\b", claim_text, re.IGNORECASE):
                    return True, "Direct contradiction: system operated with zero downtime/outage"

            # Scope limitation contradiction: Only X vs All Y
            if "only" in best_sent.lower() and re.search(r"\ball\b", claim_text, re.IGNORECASE):
                if any(w in claim_text.lower() for w in ["authorized", "permitted", "eligible", "allowed", "staff"]):
                    return True, "Scope contradiction: exclusive requirement 'only' contradicted by universal 'all'"

            if s_has_neg != c_has_neg:
                return True, f"Negation polarity conflict detected on core predicates: {list(best_matched_stems)[:4]}"

        return False, None


    @classmethod
    def check_temporal_supersession(cls, source_text: str, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if a claim refers to a contractual clause or term that was explicitly superseded.
        """
        for pattern in cls.SUPERSEDED_PATTERNS:
            match = pattern.search(source_text)
            if match:
                superseded_ref = match.group(1) if match.groups() else ""
                if superseded_ref and superseded_ref.lower() in claim_text.lower():
                    return True, f"Claim relies on clause/section '{superseded_ref}' which was explicitly superseded in source"
        return False, None

    @classmethod
    def check_comparative_conflict(cls, source_text: str, claim_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if a comparative claim (e.g. A greater than B) inverts quantitative evidence in source.
        """
        m = re.search(
            r"(\b\w+\b)\s+.*?\b(greater|larger|higher|more|lower|less|smaller|superior)\b\s+(?:reduction|increase|gain|loss|drop|change|rate)?.*?\bthan\s+(\b\w+\b)",
            claim_text,
            re.IGNORECASE
        )
        if not m:
            return False, None

        e1, comp_word, e2 = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
        s_lower = source_text.lower()
        if e1 in s_lower and e2 in s_lower:
            p1, p2 = s_lower.find(e1), s_lower.find(e2)

            def get_closest_val(pos: int) -> Optional[float]:
                matches = []
                for num_m in re.finditer(r"[-+]?\d+(?:\.\d+)?%?", s_lower):
                    raw = num_m.group(0)
                    clean = raw.replace("%", "").replace("-", "").replace("+", "")
                    try:
                        val = float(clean)
                        matches.append((abs(num_m.start() - pos), val, "%" in raw))
                    except ValueError:
                        pass
                pct_matches = [x for x in matches if x[2]]
                if pct_matches:
                    pct_matches.sort(key=lambda x: x[0])
                    return pct_matches[0][1]
                matches.sort(key=lambda x: x[0])
                return matches[0][1] if matches else None

            v1 = get_closest_val(p1)
            v2 = get_closest_val(p2)
            if v1 is not None and v2 is not None:
                if comp_word in ["greater", "larger", "higher", "more", "superior"] and v1 < v2:
                    return True, f"Comparative inversion: claim asserts {e1} > {e2}, but evidence shows {v1} vs {v2}"
                elif comp_word in ["lower", "less", "smaller"] and v1 > v2:
                    return True, f"Comparative inversion: claim asserts {e1} < {e2}, but evidence shows {v1} vs {v2}"
        return False, None

    @classmethod
    def detect_all_conflicts(cls, source_text: str, claim_text: str) -> Dict[str, Any]:
        """
        Runs all contradiction checks.
        """
        has_neg, neg_reason = cls.check_negation_conflict(source_text, claim_text)
        has_super, super_reason = cls.check_temporal_supersession(source_text, claim_text)
        has_comp, comp_reason = cls.check_comparative_conflict(source_text, claim_text)

        is_contradicted = has_neg or has_super or has_comp
        reasons = [r for r in [neg_reason, super_reason, comp_reason] if r is not None]

        return {
            "is_contradicted": is_contradicted,
            "has_negation_conflict": has_neg,
            "has_temporal_supersession": has_super,
            "has_comparative_conflict": has_comp,
            "reasons": reasons
        }
