# API

`POST /v1/systemone` accepts a state and stable question IDs. Each question has type `noul`, `choice`, or `score`; responses contain only the typed contract fields. `POST /v1/decide` accepts `query`, `state`, `decision_type`, and the relevant `options` or `rubric`.

`GET /health` returns `{"status":"ok","version":"1.0.0"}`. `GET /version` returns the V1 architecture metadata. If `APEX_API_KEY` is configured, send it in `X-API-Key` for POST endpoints.
