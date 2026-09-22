from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine


class FakeAgent:
    def predict(self, _state, questions):
        return {"answers": {"qs": {"score": 1.0, "probabilities": {"1": 1.0}, "confidence": 0.9}}, "usage": {}}


def test_score_selects_an_ordinal_rubric_level():
    result = ApexTypedDecisionEngine(agent=FakeAgent()).predict_score("Rate it", ["poor", "good", "excellent"])
    assert result.selected_level == 1
    assert result.rubric_interpretation == "good"
