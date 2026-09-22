from apex_decision.api import server


class FakeAgent:
    def predict(self, _state, questions):
        answers = {}
        for qid, question in questions.items():
            if question["type"] == "noul":
                answers[qid] = {"noul": 1.0, "confidence": 1.0, "verdict": "SUPPORTED"}
            elif question["type"] == "choice":
                option = next(iter(question["criteria"]))
                answers[qid] = {"choice": option, "probabilities": {option: 1.0}, "confidence": 1.0}
            else:
                answers[qid] = {"score": 0.0, "legend": {}, "probabilities": {"0": 1.0}, "confidence": 1.0}
        return {"answers": answers, "usage": {"total_input_tokens": 3}}


class FakeEngine:
    agent = FakeAgent()


def test_systemone_preserves_ids_and_typed_contract(monkeypatch):
    monkeypatch.setattr(server, "get_runtime", lambda: (FakeEngine(), object()))
    response = server.execute_systemone_request({"state": "evidence", "questions": {"stable": {"type": "noul", "instructions": "claim"}}})
    assert response["answers"]["stable"] == {"type": "noul", "noul": 1.0}


def test_health_and_version_are_stable():
    assert server.health() == {"status": "ok", "version": "1.0.0"}
    assert server.version()["architecture"] == "ApexStudent-50M"
