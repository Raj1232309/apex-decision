"""
ApexDecision V8 JEWEL: Complete Typed Decision Surface.
Exposes stable, typed APIs for Noul, Choice (with >=255 option support and hierarchical decomposition),
and Score (2-10 levels) with cryptographic provenance IDs and operational confidence tiering.
"""

import os
import re
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict

try:
    from .apex_student.runtime import ApexStudentAgent
except (ImportError, ValueError):
    from apex_decision.inference.apex_student.runtime import ApexStudentAgent


@dataclass
class TypedNoulDecision:
    verdict: str  # "SUPPORTED" or "NOT_SUPPORTED"
    probability: float
    confidence: float
    confidence_tier: str  # "HIGH", "MEDIUM", "LOW"
    evidence_spans: List[str]
    provenance_id: str
    reasoning_trace: List[str]
    abstained: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TypedChoiceDecision:
    selected_option: str
    probabilities: Dict[str, float]
    confidence: float
    evidence_spans: List[str]
    provenance_id: str
    verification_state: str  # "VERIFIED", "UNVERIFIED"
    abstained: bool
    hierarchical_stages_used: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TypedScoreDecision:
    selected_level: int
    continuous_score: float
    probabilities: Dict[str, float]
    confidence: float
    rubric_interpretation: str
    evidence_spans: List[str]
    provenance_id: str
    abstained: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ApexTypedDecisionEngine:
    """
    JEWEL typed decision surface for ApexDecision V8.
    Encapsulates high-velocity parallel decision runtime into stable typed contracts.
    """

    def __init__(self, agent: Optional[ApexStudentAgent] = None, device: Optional[str] = None):
        self.agent = agent or ApexStudentAgent(device=device)

    @staticmethod
    def generate_provenance_id(state: str, question: str, ans_str: str) -> str:
        """Deterministic SHA-256 provenance signature."""
        raw = f"{state.strip()}|{question.strip()}|{ans_str.strip()}".encode("utf-8")
        return f"prov_{hashlib.sha256(raw).hexdigest()[:16]}"

    def predict_noul(
        self,
        question: str,
        state: Union[str, dict, list],
        metadata: Optional[Dict[str, Any]] = None
    ) -> TypedNoulDecision:
        """Execute typed binary belief grounding."""
        state_str = state if isinstance(state, str) else json.dumps(state)
        q_dict = {
            "q0": {
                "type": "noul",
                "instructions": question
            }
        }
        res = self.agent.predict(state_str, q_dict)
        ans = res["answers"]["q0"]
        p = float(ans["noul"])
        conf = float(ans["confidence"])
        
        tier = "HIGH" if conf >= 0.90 else ("MEDIUM" if conf >= 0.70 else "LOW")
        prov_id = self.generate_provenance_id(state_str, question, ans["verdict"])
        
        trace = [
            f"Evaluated statement: '{question}'",
            f"Factual consistency: {ans.get('numeric_consistent', True)}",
            f"Belief probability: {p:.4f} (Confidence: {conf:.4f}, Tier: {tier})"
        ]

        return TypedNoulDecision(
            verdict=ans["verdict"],
            probability=p,
            confidence=conf,
            confidence_tier=tier,
            evidence_spans=ans.get("evidence_spans", []),
            provenance_id=prov_id,
            reasoning_trace=trace,
            abstained=bool(ans.get("abstained", False))
        )

    def predict_choice(
        self,
        question: str,
        options: Union[List[str], Dict[str, str]],
        state: Union[str, dict, list] = ""
    ) -> TypedChoiceDecision:
        """
        Execute typed option choice selection.
        Supports >=255 options. If option count exceeds 255, automatically
        applies hierarchical clustering decomposition.
        """
        state_str = state if isinstance(state, str) else json.dumps(state)
        if isinstance(options, list):
            crit = {opt: opt for opt in options}
        else:
            crit = options

        num_opts = len(crit)
        hierarchical_stages = 1

        # Hierarchical decomposition for > 255 options
        if num_opts > 255:
            hierarchical_stages = 2
            keys = list(crit.keys())
            batch_size = 64
            num_batches = (num_opts + batch_size - 1) // batch_size
            
            # Stage 1: Macro-batch selection
            macro_options = {}
            for b in range(num_batches):
                b_keys = keys[b * batch_size : (b + 1) * batch_size]
                macro_label = f"Group_{b+1}: {b_keys[0]} to {b_keys[-1]}"
                macro_options[macro_label] = f"Options range containing {', '.join(b_keys[:3])}..."

            stage1_q = {
                "stage1": {
                    "type": "choice",
                    "instructions": f"Identify the target group for: {question}",
                    "criteria": macro_options
                }
            }
            s1_res = self.agent.predict(state_str, stage1_q)
            s1_choice = s1_res["answers"]["stage1"]["choice"]
            
            # Extract chosen group index
            m = re.search(r"Group_(\d+)", str(s1_choice))
            b_idx = int(m.group(1)) - 1 if m else 0
            b_idx = max(0, min(b_idx, num_batches - 1))
            
            selected_sub_keys = keys[b_idx * batch_size : (b_idx + 1) * batch_size]
            crit = {k: crit[k] for k in selected_sub_keys}

        q_dict = {
            "qc": {
                "type": "choice",
                "instructions": question,
                "criteria": crit
            }
        }
        res = self.agent.predict(state_str, q_dict)
        ans = res["answers"]["qc"]
        
        prov_id = self.generate_provenance_id(state_str, question, str(ans["choice"]))

        return TypedChoiceDecision(
            selected_option=ans["choice"],
            probabilities=ans["probabilities"],
            confidence=float(ans["confidence"]),
            evidence_spans=ans.get("evidence_spans", []),
            provenance_id=prov_id,
            verification_state="VERIFIED" if ans.get("zero_invention_verified", True) else "UNVERIFIED",
            abstained=bool(ans.get("abstained", False)),
            hierarchical_stages_used=hierarchical_stages
        )

    def predict_score(
        self,
        question: str,
        rubric: List[str],
        state: Union[str, dict, list] = ""
    ) -> TypedScoreDecision:
        """
        Execute typed score evaluation over 2 to 10 ordinal rubric tiers.
        """
        state_str = state if isinstance(state, str) else json.dumps(state)
        q_dict = {
            "qs": {
                "type": "score",
                "instructions": question,
                "criteria": rubric
            }
        }
        res = self.agent.predict(state_str, q_dict)
        ans = res["answers"]["qs"]
        
        continuous_score = float(ans["score"])
        selected_level = min(len(rubric) - 1, max(0, int(round(continuous_score))))
        rubric_interpretation = rubric[selected_level]
        
        prov_id = self.generate_provenance_id(state_str, question, str(selected_level))

        return TypedScoreDecision(
            selected_level=selected_level,
            continuous_score=continuous_score,
            probabilities=ans["probabilities"],
            confidence=float(ans["confidence"]),
            rubric_interpretation=rubric_interpretation,
            evidence_spans=ans.get("evidence_spans", []),
            provenance_id=prov_id,
            abstained=bool(ans.get("abstained", False))
        )
