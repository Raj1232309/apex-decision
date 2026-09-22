from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine


class FakeAgent:
    def predict(self, _state, questions):
        key = next(iter(questions))
        options = questions[key]["criteria"]
        choice = next(iter(options))
        return {"answers": {key: {"choice": choice, "probabilities": {choice: 1.0}, "confidence": 1.0}}, "usage": {}}


def test_choice_is_restricted_to_supplied_options():
    result = ApexTypedDecisionEngine(agent=FakeAgent()).predict_choice("Select priority", ["low", "high"])
    assert result.selected_option in {"low", "high"}
    assert result.hierarchical_stages_used == 1
