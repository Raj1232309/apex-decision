from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine


class FakeAgent:
    def predict(self, state, questions):
        return {"answers": {"q0": {"noul": 1.0, "confidence": 0.95, "verdict": "SUPPORTED", "evidence_spans": [state]}}, "usage": {}}


def test_noul_returns_typed_grounded_decision():
    result = ApexTypedDecisionEngine(agent=FakeAgent()).predict_noul("Paris is France's capital.", "Paris is France's capital.")
    assert result.verdict == "SUPPORTED"
    assert result.confidence_tier == "HIGH"
    assert result.provenance_id.startswith("prov_")
