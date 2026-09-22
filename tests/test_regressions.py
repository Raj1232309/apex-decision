from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine


def test_provenance_identifier_is_deterministic():
    assert ApexTypedDecisionEngine.generate_provenance_id("state", "question", "answer") == ApexTypedDecisionEngine.generate_provenance_id("state", "question", "answer")
