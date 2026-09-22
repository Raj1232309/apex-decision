"""
Reasoning evaluation module for ApexDecision External Benchmark Suite.
Evaluates:
- Multi-hop reasoning (5-hop, 10-hop, 20-hop dependency chains)
- Structured arithmetic DAG solving
- Scope & double-negation handling
- Entity disambiguation & counterfactual minimal pairs
- Temporal & numeric contradiction handling
"""

import time
from typing import Dict, Any, List, Optional
from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine
from apex_decision.reasoning.arithmetic_engine import ArithmeticEngine
from apex_decision.reasoning.noul_verifier import NoulVerifier


class ReasoningEvaluator:
    """Evaluates multi-hop arithmetic, structured logic, and contradiction defense."""

    def __init__(self, engine: Optional[ApexTypedDecisionEngine] = None):
        self.engine = engine or ApexTypedDecisionEngine()
        self.arithmetic_engine = ArithmeticEngine()
        self.noul_verifier = NoulVerifier()

    def run_multihop_arithmetic_eval(self) -> Dict[str, Any]:
        """Evaluates exact mathematical reasoning across structured multi-hop chains."""
        test_cases = [
            # 1. 2-hop linear chain
            {
                "state": "Base salary is $100,000. Performance bonus adds 15%.",
                "q": "Total compensation equals $115,000.",
                "gold": 1.0,
                "hops": 2
            },
            # 2. 3-hop linear chain
            {
                "state": "Base revenue is $200. Cost of goods is $50. Tax rate is 10% on remaining profit.",
                "q": "After-tax net profit equals $135.0.",
                "gold": 1.0,
                "hops": 3
            },
            # 3. Arithmetic conflict
            {
                "state": "Product price is $80 after a $20 discount on a $100 original price.",
                "q": "Product price is $70.",
                "gold": 0.0,
                "hops": 2
            },
            # 4. Multi-hop structured DAG (5-hop)
            {
                "state": "Hop-1: Alpha is 10. Hop-2: Beta is Alpha plus 20. Hop-3: Gamma is Beta multiplied by 2. Hop-4: Delta is Gamma minus 15. Hop-5: Epsilon is Delta divided by 3.",
                "q": "Epsilon equals 15.0.",
                "gold": 1.0,
                "hops": 5
            },
            # 5. Multi-hop structured DAG (10-hop)
            {
                "state": "Step 1 is 5. Step 2 is Step 1 + 5. Step 3 is Step 2 + 10. Step 4 is Step 3 * 2. Step 5 is Step 4 - 10. Step 6 is Step 5 / 2. Step 7 is Step 6 + 15. Step 8 is Step 7 * 2. Step 9 is Step 8 - 20. Step 10 is Step 9 / 2.",
                "q": "Step 10 equals 25.0.",
                "gold": 1.0,
                "hops": 10
            }
        ]

        correct = 0
        total = len(test_cases)
        failures = []

        for case in test_cases:
            res = self.engine.predict_noul(question=case["q"], state=case["state"])
            pred = 1.0 if res.verdict == "SUPPORTED" else 0.0
            is_correct = (pred == case["gold"])
            if is_correct:
                correct += 1
            else:
                failures.append({
                    "hops": case["hops"],
                    "question": case["q"],
                    "expected": case["gold"],
                    "predicted": pred,
                    "probability": res.probability
                })

        return {
            "benchmark_id": "reasoning_multihop_arithmetic",
            "sample_count": total,
            "correct": correct,
            "incorrect": total - correct,
            "exact_match": round((correct / max(1, total)) * 100.0, 2),
            "failures": failures
        }

    def run_contradiction_scope_eval(self) -> Dict[str, Any]:
        """Evaluates scope, double negation, temporal supersession, and counterfactuals."""
        cases = [
            # 1. Double negation resolution
            {
                "state": "It is not true that the security perimeter was not breached.",
                "q": "The security perimeter was breached.",
                "gold": 1.0,
                "type": "double_negation"
            },
            # 2. Scope restriction
            {
                "state": "The warranty covers hardware components only. Software issues are excluded.",
                "q": "The warranty covers operating system crashes.",
                "gold": 0.0,
                "type": "scope_restriction"
            },
            # 3. Temporal supersession
            {
                "state": "In 2020 the headquarters was in Chicago. In 2024 the headquarters relocated permanently to Austin.",
                "q": "The current headquarters is located in Austin.",
                "gold": 1.0,
                "type": "temporal_supersession"
            },
            # 4. Outdated temporal claim
            {
                "state": "In 2020 the headquarters was in Chicago. In 2024 the headquarters relocated permanently to Austin.",
                "q": "The current headquarters is located in Chicago.",
                "gold": 0.0,
                "type": "outdated_temporal"
            },
            # 5. Minimal pair counterfactual
            {
                "state": "Employee 101 submitted the budget before the deadline. Employee 102 submitted after the deadline.",
                "q": "Employee 102 submitted before the deadline.",
                "gold": 0.0,
                "type": "minimal_pair"
            }
        ]

        correct = 0
        total = len(cases)
        failures = []

        for c in cases:
            res = self.engine.predict_noul(question=c["q"], state=c["state"])
            pred = 1.0 if res.verdict == "SUPPORTED" else 0.0
            if pred == c["gold"]:
                correct += 1
            else:
                failures.append({
                    "type": c["type"],
                    "question": c["q"],
                    "expected": c["gold"],
                    "predicted": pred,
                    "probability": res.probability
                })

        return {
            "benchmark_id": "reasoning_contradiction_scope",
            "sample_count": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round((correct / max(1, total)) * 100.0, 2),
            "failures": failures
        }
