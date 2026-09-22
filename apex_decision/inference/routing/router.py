"""
ApexDecision V7 Dynamic Specialist Router.
Directs queries to FAST_PATH (<10ms), DEEP_PATH (multi-hop / long context),
or SAFETY_PATH (deterministic provenance & contradiction checks).
"""

from typing import Dict, Any, Optional, Tuple, List
import re
from enum import Enum


class RouteType(str, Enum):
    FAST_PATH = "FAST_PATH"
    DEEP_PATH = "DEEP_PATH"
    SAFETY_PATH = "SAFETY_PATH"


class ApexRouter:
    """
    Deterministic adaptive compute allocator for ApexDecision V7.
    """

    NUMERIC_REGEX = re.compile(r"[-+]?\$?\d+(?:\.\d+)?%?")
    CONTRADICTION_TERMS = {"superseded", "amendment", "neither", "nor", "void", "denied", "prohibited"}

    @classmethod
    def route(
        cls,
        state: str,
        question_ins: str,
        qtype: str = "noul",
        options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Determines execution route based on input complexity, token length, and risk.
        """
        word_count = len(state.split())

        # 1. Check for Long Context requirement
        if word_count > 1024:
            return {
                "route": RouteType.DEEP_PATH,
                "reason": f"Long-context state ({word_count} words > 1024), invoking Hierarchical Engine",
                "estimated_cost": "moderate"
            }

        # 2. Check for High-Cardinality Choice
        if qtype.lower() == "choice" and options and len(options) > 20:
            return {
                "route": RouteType.DEEP_PATH,
                "reason": f"High-cardinality choice space ({len(options)} options > 20), invoking 2-stage hierarchical head",
                "estimated_cost": "moderate"
            }

        # 3. Check for Numeric or Contradiction Risk
        has_numbers = bool(cls.NUMERIC_REGEX.search(question_ins))
        has_contradiction = bool(cls.CONTRADICTION_TERMS & set(re.findall(r"\b\w+\b", question_ins.lower())))

        if has_numbers or has_contradiction:
            return {
                "route": RouteType.SAFETY_PATH,
                "reason": "Numeric expressions or contract supersession keywords detected, engaging Safety Validators",
                "estimated_cost": "fast+safe"
            }

        # 4. Standard Fast Path
        return {
            "route": RouteType.FAST_PATH,
            "reason": "Standard single-pass evaluation within local sequence budget",
            "estimated_cost": "minimal (<10ms)"
        }
