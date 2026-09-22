"""
Independent Grounding, Attribution, Hallucination & Jev Typed Decision Benchmark.
Evaluates:
- Unseen documents, entities, aliases
- Conflicting & distractor evidence
- Missing/negative evidence refusal
- Zero-invention span attribution
- Jev-style typed decision wire contract validation
"""

import time
from typing import Dict, Any, List, Optional
from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine
from apex_decision.verification.provenance_verifier import ProvenanceVerifier


class CoreEvaluator:
    """Evaluates grounding, attribution, hallucination defense, and typed decisions."""

    def __init__(self, engine: Optional[ApexTypedDecisionEngine] = None):
        self.engine = engine or ApexTypedDecisionEngine()
        self.verifier = ProvenanceVerifier()

    def run_grounding_eval(self) -> Dict[str, Any]:
        """Independent grounding & attribution across unseen entities & distractors."""
        test_cases = [
            # 1. Unseen entity direct entailment
            {
                "state": "The Zephyr-9 drone is engineered by Aether Dynamics in Reykjavik.",
                "q": "The Zephyr-9 drone is manufactured by Aether Dynamics.",
                "gold": 1.0,
                "category": "unseen_entity_direct"
            },
            # 2. Unseen entity alias resolution
            {
                "state": "Apex Corp, known globally as AC Industries, acquired Helios Lab in 2024.",
                "q": "AC Industries acquired Helios Lab in 2024.",
                "gold": 1.0,
                "category": "unseen_alias"
            },
            # 3. Conflicting evidence (near-miss contradiction)
            {
                "state": "Project Falcon launch occurred on October 12, 2025.",
                "q": "Project Falcon launch occurred on October 14, 2025.",
                "gold": 0.0,
                "category": "conflicting_evidence"
            },
            # 4. Missing evidence (unsupported claim refusal)
            {
                "state": "The library houses over 50,000 manuscripts dating from the 14th century.",
                "q": "The library contains rare Galileo Galilei original letters.",
                "gold": 0.0,
                "category": "missing_evidence"
            },
            # 5. Negative evidence
            {
                "state": "Strictly no third-party plugins are permitted within the core kernel.",
                "q": "Third-party plugins can run directly in the core kernel.",
                "gold": 0.0,
                "category": "negative_evidence"
            },
            # 6. Distractor evidence
            {
                "state": "Entity Alpha reported $5M profit. Entity Beta reported $20M loss. Entity Gamma was dormant.",
                "q": "Entity Beta reported $5M profit.",
                "gold": 0.0,
                "category": "distractor_evidence"
            },
            # 7. Unseen document entity attribution
            {
                "state": "NeuroLink-X completed Phase III trials with 94.2% efficacy in Stockholm.",
                "q": "NeuroLink-X completed Phase III trials with 94.2% efficacy.",
                "gold": 1.0,
                "category": "unseen_doc_attribution"
            },
            # 8. Numeric boundary constraint
            {
                "state": "Maximum operating temperature must remain strictly below 85.0°C.",
                "q": "The system is authorized to operate at 89.5°C.",
                "gold": 0.0,
                "category": "numeric_boundary"
            }
        ]

        correct = 0
        total = len(test_cases)
        failures = []
        span_checks_passed = 0
        span_checks_total = 0

        for case in test_cases:
            res = self.engine.predict_noul(question=case["q"], state=case["state"])
            pred = 1.0 if res.verdict == "SUPPORTED" else 0.0
            is_correct = (pred == case["gold"])

            # Verify evidence spans for zero fabrication
            for sp in res.evidence_spans:
                span_checks_total += 1
                if self.verifier.verify_evidence_span(case["state"], sp)["verified"]:
                    span_checks_passed += 1

            if is_correct:
                correct += 1
            else:
                failures.append({
                    "category": case["category"],
                    "question": case["q"],
                    "expected": case["gold"],
                    "predicted": pred,
                    "probability": res.probability,
                    "confidence": res.confidence
                })

        acc = (correct / max(1, total)) * 100.0
        return {
            "benchmark_id": "grounding_unseen_attribution",
            "sample_count": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(acc, 2),
            "fabricated_spans": span_checks_total - span_checks_passed,
            "span_verification_rate": 100.0 if span_checks_total == 0 else round((span_checks_passed / span_checks_total) * 100.0, 2),
            "failures": failures
        }

    def run_hallucination_eval(self) -> Dict[str, Any]:
        """Evaluates refusal and non-invention across unsupported claims and invented options."""
        cases = [
            # 1. Unsupported entity
            {"state": "The treaty was signed in Geneva in 1949.", "q": "The treaty was signed by Albert Einstein.", "gold_refuse": True},
            # 2. Fabricated metric
            {"state": "Model latency is 14ms.", "q": "Model latency is 0.001ms.", "gold_refuse": True},
            # 3. Passing mention false positive entrapment
            {"state": "The committee reviewed a rejected proposal regarding automated sanctions.", "q": "Automated sanctions were enacted by the committee.", "gold_refuse": True},
            # 4. Valid supported claim
            {"state": "Titan is the largest moon of Saturn.", "q": "Titan is the largest moon of Saturn.", "gold_refuse": False}
        ]

        correct_refusals = 0
        total = len(cases)
        failures = []

        for c in cases:
            res = self.engine.predict_noul(question=c["q"], state=c["state"])
            refused = (res.verdict == "NOT_SUPPORTED")
            is_correct = (refused == c["gold_refuse"])
            if is_correct:
                correct_refusals += 1
            else:
                failures.append({
                    "question": c["q"],
                    "expected_refusal": c["gold_refuse"],
                    "verdict": res.verdict,
                    "probability": res.probability
                })

        return {
            "benchmark_id": "hallucination_refusal",
            "sample_count": total,
            "correct": correct_refusals,
            "incorrect": total - correct_refusals,
            "refusal_precision": round((correct_refusals / max(1, total)) * 100.0, 2),
            "failures": failures
        }

    def run_typed_decisions_eval(self) -> Dict[str, Any]:
        """Validates Jev-style typed output schema contracts."""
        # 1. Noul exact contract
        noul_res = self.engine.predict_noul(
            question="Water boils at 100°C at 1 atm.",
            state="Water boils at 100°C under 1 atmosphere of pressure."
        )
        assert hasattr(noul_res, "verdict")
        assert hasattr(noul_res, "probability")
        assert hasattr(noul_res, "confidence")
        assert hasattr(noul_res, "provenance_id")

        # 2. Choice exact contract
        choice_res = self.engine.predict_choice(
            question="Select primary cloud provider.",
            options={"AWS": "Amazon Web Services", "GCP": "Google Cloud Platform", "AZURE": "Microsoft Azure"},
            state="All storage buckets reside in GCP us-central1."
        )
        assert choice_res.selected_option == "GCP"
        assert set(choice_res.probabilities.keys()) == {"AWS", "GCP", "AZURE"}

        # 3. Score exact contract
        score_res = self.engine.predict_score(
            question="Rate compliance level.",
            rubric=["Non-compliant", "Partially compliant", "Fully compliant"],
            state="All audits passed with zero non-conformances. Fully compliant."
        )
        assert score_res.selected_level == 2
        assert score_res.rubric_interpretation == "Fully compliant"

        return {
            "benchmark_id": "typed_decisions_jev_contract",
            "sample_count": 3,
            "correct": 3,
            "incorrect": 0,
            "schema_compliance": 100.0,
            "failures": []
        }
