"""Small, intentional client surface for ApexDecision."""

from __future__ import annotations

from typing import Any, Sequence


class ApexClient:
    """Call a local ApexDecision server, or use an in-memory engine when supplied."""

    def __init__(self, endpoint: str | None = "http://127.0.0.1:8000", *, engine: Any = None, api_key: str | None = None, timeout: float = 30.0):
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.engine = engine
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.endpoint:
            raise RuntimeError("An endpoint is required unless an in-memory engine is provided.")
        import httpx
        response = httpx.post(f"{self.endpoint}/v1/decide", json=payload, headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def noul(self, statement: str, state: str = "") -> Any:
        if self.engine:
            return self.engine.predict_noul(question=statement, state=state)
        return self._request({"query": statement, "state": state, "decision_type": "noul"})

    def choice(self, question: str, options: Sequence[str], state: str = "") -> Any:
        if self.engine:
            return self.engine.predict_choice(question=question, options=list(options), state=state)
        return self._request({"query": question, "state": state, "decision_type": "choice", "options": list(options)})

    def score(self, question: str, criteria: Sequence[str], state: str = "") -> Any:
        if self.engine:
            return self.engine.predict_score(question=question, rubric=list(criteria), state=state)
        return self._request({"query": question, "state": state, "decision_type": "score", "rubric": list(criteria)})

    def orchestrate(self, query: str, state: str = "", *, allow_user_confirmation: bool = False) -> dict[str, Any]:
        return self._request({"query": query, "state": state, "decision_type": "orchestrate", "allow_user_confirmation": allow_user_confirmation})
