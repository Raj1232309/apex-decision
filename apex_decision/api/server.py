"""Production HTTP API for ApexDecision V1."""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Union
from collections import deque

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from apex_decision import __version__
from apex_decision.config import Settings
from apex_decision.inference.typed_decision_engine import ApexTypedDecisionEngine
from apex_decision.tools.orchestrator import ApexToolOrchestrator

logger = logging.getLogger("apex_decision.api")
app = FastAPI(title="ApexDecision API", version=__version__)

# In-memory audit ring-buffer for internal compliance and provenance tracking
_AUDIT_STORE: deque = deque(maxlen=1000)

engine: ApexTypedDecisionEngine | None = None
orchestrator: ApexToolOrchestrator | None = None


def get_runtime() -> tuple[ApexTypedDecisionEngine, ApexToolOrchestrator]:
    """Initialize the model lazily so health checks do not consume model memory."""
    global engine, orchestrator
    if engine is None:
        engine = ApexTypedDecisionEngine(device=Settings.from_env().device)
        orchestrator = ApexToolOrchestrator(decision_engine=engine)
    return engine, orchestrator  # type: ignore[return-value]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured_key = Settings.from_env().api_key
    if configured_key and x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Authentication failed.")


@app.exception_handler(Exception)
async def safe_unexpected_error(_: Request, exc: Exception):
    logger.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content={"type": "about:blank", "title": "Internal Server Error", "status": 500, "detail": "The request could not be completed."},
        media_type="application/problem+json",
    )


# =====================================================================
# Request & Response Models for Jev-Style SystemOne API
# =====================================================================

class SystemOneQuestion(BaseModel):
    type: str  # "noul", "choice", "score"
    instructions: str
    options: Optional[Union[List[str], Dict[str, str]]] = None
    criteria: Optional[Union[List[str], Dict[str, str]]] = None
    rubric: Optional[Union[List[str], Dict[str, str]]] = None
    legend: Optional[Union[List[str], Dict[str, str]]] = None


class SystemOneRequest(BaseModel):
    state: str = ""
    model: Optional[str] = "apex-50m-v8"
    questions: Dict[str, Any]


# =====================================================================
# Request Model for Native Apex API
# =====================================================================

class DecisionRequest(BaseModel):
    query: str
    state: Optional[str] = ""
    decision_type: Optional[str] = "orchestrate"  # "orchestrate", "noul", "choice", "score"
    options: Optional[Union[List[str], Dict[str, str]]] = None
    rubric: Optional[List[str]] = None
    allow_user_confirmation: Optional[bool] = False


# =====================================================================
# Core Jev Compatibility Logic
# =====================================================================

def execute_systemone_request(req_data: Dict[str, Any]) -> Dict[str, Any]:
    state_text = req_data.get("state", "")
    req_model = req_data.get("model", "apex-50m-v8")
    raw_questions = req_data.get("questions", {})

    if not isinstance(raw_questions, dict) or not raw_questions:
        raise HTTPException(status_code=400, detail="'questions' must be a non-empty dictionary of questions.")

    # Normalize question format for the student runtime
    prep_questions = {}
    for qid, qdef in raw_questions.items():
        if not isinstance(qdef, dict):
            continue
        q_type = str(qdef.get("type", "noul")).lower().strip()
        ins = qdef.get("instructions", "")

        if q_type == "choice":
            raw_opts = qdef.get("options") or qdef.get("criteria")
            if isinstance(raw_opts, list):
                crit = {opt: opt for opt in raw_opts}
            elif isinstance(raw_opts, dict):
                crit = raw_opts
            else:
                crit = {}
            prep_questions[qid] = {
                "type": "choice",
                "instructions": ins,
                "criteria": crit
            }
        elif q_type == "score":
            raw_crit = qdef.get("criteria") or qdef.get("rubric") or qdef.get("legend") or qdef.get("options")
            if isinstance(raw_crit, list):
                crit = raw_crit
            elif isinstance(raw_crit, dict):
                try:
                    crit = [raw_crit[str(i)] for i in range(len(raw_crit))]
                except Exception:
                    crit = list(raw_crit.values())
            else:
                crit = []
            prep_questions[qid] = {
                "type": "score",
                "instructions": ins,
                "criteria": crit
            }
        else:  # noul
            prep_questions[qid] = {
                "type": "noul",
                "instructions": ins,
                "criteria": qdef.get("criteria")
            }

    engine, _ = get_runtime()
    # Execute single forward pass via ApexStudentAgent
    raw_res = engine.agent.predict(state_text, prep_questions)
    raw_answers = raw_res.get("answers", {})

    # Filter into exact Jev wire shapes with zero extra fields
    jev_answers = {}
    for qid in raw_questions.keys():
        ans = raw_answers.get(qid, {})
        q_type = str(raw_questions[qid].get("type", "noul")).lower().strip()

        if q_type == "noul":
            # EXACT Jev shape: {"type": "noul", "noul": float}
            p_noul = float(ans.get("noul", 0.5))
            jev_answers[qid] = {
                "type": "noul",
                "noul": round(p_noul, 4)
            }
        elif q_type == "choice":
            # EXACT Jev shape: {"type": "choice", "choice": str, "probabilities": dict, "confidence": float}
            probs = ans.get("probabilities", {})
            jev_answers[qid] = {
                "type": "choice",
                "choice": str(ans.get("choice", "")),
                "probabilities": {str(k): round(float(v), 4) for k, v in probs.items()},
                "confidence": round(float(ans.get("confidence", 0.5)), 4)
            }
        elif q_type == "score":
            # EXACT Jev shape: {"type": "score", "score": float, "legend": dict, "probabilities": dict, "confidence": float}
            legend = ans.get("legend", {})
            if not legend:
                crit_list = prep_questions[qid].get("criteria", [])
                legend = {str(i): str(c) for i, c in enumerate(crit_list)}
            else:
                legend = {str(k): str(v) for k, v in legend.items()}

            probs = ans.get("probabilities", {})
            jev_answers[qid] = {
                "type": "score",
                "score": round(float(ans.get("score", 0.0)), 4),
                "legend": legend,
                "probabilities": {str(k): round(float(v), 4) for k, v in probs.items()},
                "confidence": round(float(ans.get("confidence", 0.5)), 4)
            }

    input_tokens = int(raw_res.get("usage", {}).get("total_input_tokens", 0))
    top_response = {
        "model": req_model or "apex-50m-v8",
        "answers": jev_answers,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": 0
        }
    }

    # Store full execution trace internally for auditing
    _AUDIT_STORE.append({
        "timestamp": time.time(),
        "model": req_model,
        "state": state_text,
        "raw_questions": raw_questions,
        "raw_answers": raw_answers,
        "usage": raw_res.get("usage", {})
    })

    return top_response


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/version")
def version():
    return {"version": __version__, "architecture": "ApexStudent-50M"}


@app.post("/v1/systemone")
async def systemone_endpoint(request: Request, _: None = Header(default=None, alias="X-API-Key")):
    """
    Jev-compatible SystemOne endpoint.
    Accepts arbitrary JSON conforming to Jev's documented contract.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    require_api_key(_)
    return execute_systemone_request(body)


@app.post("/v1/systemone/compatibility-test")
def systemone_compatibility_test(x_api_key: str | None = Header(default=None)):
    """
    Executes a comprehensive local contract test verifying that /v1/systemone
    outputs match Jev's documented wire shapes with zero disallowed fields.
    """
    require_api_key(x_api_key)
    test_results = []

    # Test 1: Noul contract
    noul_req = {
        "state": "The speed of light in vacuum is approximately 300,000 km/s.",
        "model": "apex-50m-v8",
        "questions": {
            "q_test_noul": {
                "type": "noul",
                "instructions": "The speed of light in vacuum is approximately 300,000 km/s."
            }
        }
    }
    resp1 = execute_systemone_request(noul_req)
    assert set(resp1.keys()) == {"model", "answers", "usage"}, "Top-level keys mismatch"
    assert set(resp1["usage"].keys()) == {"input_tokens", "output_tokens"}, "Usage keys mismatch"
    noul_ans = resp1["answers"].get("q_test_noul")
    assert noul_ans is not None, "Missing question answer"
    assert set(noul_ans.keys()) == {"type", "noul"}, f"Noul answer contains disallowed keys: {set(noul_ans.keys())}"
    assert isinstance(noul_ans["noul"], (int, float)), "Noul value must be a number"
    test_results.append({
        "test": "noul_schema_contract",
        "status": "pass",
        "sample": noul_ans
    })

    # Test 2: Choice contract
    choice_req = {
        "state": "Tokyo is the capital city of Japan.",
        "model": "apex-50m-v8",
        "questions": {
            "q_test_choice": {
                "type": "choice",
                "instructions": "What is the capital of Japan?",
                "options": ["Tokyo", "Kyoto", "Osaka"]
            }
        }
    }
    resp2 = execute_systemone_request(choice_req)
    choice_ans = resp2["answers"].get("q_test_choice")
    assert choice_ans is not None, "Missing choice answer"
    assert set(choice_ans.keys()) == {"type", "choice", "probabilities", "confidence"}, f"Choice contains disallowed keys: {set(choice_ans.keys())}"
    assert choice_ans["choice"] in ["Tokyo", "Kyoto", "Osaka"], "Selected choice must be in options"
    assert isinstance(choice_ans["probabilities"], dict), "Probabilities must be a dictionary"
    assert isinstance(choice_ans["confidence"], (int, float)), "Confidence must be a number"
    test_results.append({
        "test": "choice_schema_contract",
        "status": "pass",
        "sample": choice_ans
    })

    # Test 3: Score contract
    score_req = {
        "state": "The service response time was under 20 milliseconds.",
        "model": "apex-50m-v8",
        "questions": {
            "q_test_score": {
                "type": "score",
                "instructions": "Rate service responsiveness.",
                "criteria": ["Slow", "Acceptable", "Ultra-fast"]
            }
        }
    }
    resp3 = execute_systemone_request(score_req)
    score_ans = resp3["answers"].get("q_test_score")
    assert score_ans is not None, "Missing score answer"
    assert set(score_ans.keys()) == {"type", "score", "legend", "probabilities", "confidence"}, f"Score contains disallowed keys: {set(score_ans.keys())}"
    assert isinstance(score_ans["score"], (int, float)), "Score must be a number"
    assert isinstance(score_ans["legend"], dict), "Legend must be a dict"
    assert isinstance(score_ans["probabilities"], dict), "Probabilities must be a dict"
    assert set(score_ans["legend"].keys()) == {"0", "1", "2"}, "Legend must have string indices"
    test_results.append({
        "test": "score_schema_contract",
        "status": "pass",
        "sample": score_ans
    })

    # Test 4: Batch ID preservation & cross-talk prevention
    batch_req = {
        "state": "Entity A operates in region X. Entity B operates in region Y.",
        "model": "apex-50m-v8",
        "questions": {
            "custom_id_101": {
                "type": "noul",
                "instructions": "Entity A operates in region X."
            },
            "custom_id_202": {
                "type": "choice",
                "instructions": "Which entity operates in region Y?",
                "options": ["Entity A", "Entity B"]
            },
            "custom_id_303": {
                "type": "score",
                "instructions": "Score coverage level.",
                "criteria": ["None", "Partial", "Full"]
            }
        }
    }
    resp4 = execute_systemone_request(batch_req)
    assert set(resp4["answers"].keys()) == {"custom_id_101", "custom_id_202", "custom_id_303"}, "Question IDs were not preserved"
    test_results.append({
        "test": "batch_id_preservation",
        "status": "pass",
        "question_ids_received": list(resp4["answers"].keys())
    })

    return {
        "status": "passed",
        "contract": "jev_systemone_wire_compatible",
        "total_tests": len(test_results),
        "passed_tests": len(test_results),
        "failed_tests": 0,
        "details": test_results
    }


# =====================================================================
# Native Decision & Orchestration Endpoints
# =====================================================================

@app.post("/decide")
@app.post("/v1/decide")
def decide(req: DecisionRequest, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    engine, orchestrator = get_runtime()
    dtype = (req.decision_type or "orchestrate").lower()
    try:
        if dtype == "noul":
            return engine.predict_noul(question=req.query, state=req.state or "").to_dict()
        elif dtype == "choice":
            if not req.options:
                raise HTTPException(status_code=400, detail="'options' field required for choice decisions")
            return engine.predict_choice(question=req.query, options=req.options, state=req.state or "").to_dict()
        elif dtype == "score":
            if not req.rubric:
                raise HTTPException(status_code=400, detail="'rubric' field required for score decisions")
            return engine.predict_score(question=req.query, rubric=req.rubric, state=req.state or "").to_dict()
        else:  # "orchestrate" or tool-calling request
            return orchestrator.run(
                user_query=req.query,
                context_state=req.state or "",
                decision_type="noul",
                allow_user_confirmation=req.allow_user_confirmation or False
            ).to_dict()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Decision request failed")
        raise HTTPException(status_code=500, detail="The decision could not be completed.")


def start_server(host: Optional[str] = None, port: Optional[int] = None):
    settings = Settings.from_env()
    uvicorn.run(app, host=host or settings.host, port=port or settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    start_server()
