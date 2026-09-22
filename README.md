# ApexDecision

ApexDecision is a local, typed decision engine built around Noul (belief grounding), Choice, Score, and permission-gated tool orchestration. Version 1 ships with the ApexStudent-50M checkpoint.

## Install

```bash
python -m pip install -e .
```

The repository includes the production checkpoint. To store it elsewhere, set `APEX_MODEL_PATH` to its `student-50m-v8.pt` path. Copy `.env.example` to `.env` and configure the host, port, device, or optional API key as needed.

## Run the local API

```bash
apex-decision
```

The default endpoint is `http://127.0.0.1:8000`. `GET /health` and `GET /version` never load the model. Use `POST /v1/systemone` for the Jev-compatible typed contract and `POST /v1/decide` for native typed decisions and tool orchestration.

```python
from apex_decision import ApexClient

client = ApexClient()
result = client.noul("Paris is the capital of France.", "Paris is France's capital.")
```

## Independent External Benchmark Suite

ApexDecision includes a fully independent, reproducible benchmark suite in `benchmark/`:

```bash
# Run the complete external benchmark suite
python -m benchmark.run_all

# Run quick verification mode
python -m benchmark.run_all --quick

# Run specific evaluation suites
python -m benchmark.run_all --suite core
python -m benchmark.run_all --suite reasoning
python -m benchmark.run_all --suite tools
python -m benchmark.run_all --suite multilingual
python -m benchmark.run_all --suite long_context
python -m benchmark.run_all --suite performance
```

### Launch Gradio Benchmark UI & Telemetry Center

```bash
python benchmark/app.py
```

Runs an interactive dashboard on `http://127.0.0.1:7860` with real-time log streaming, system telemetry, and results downloads. Also containerized via `Dockerfile` for Hugging Face Docker Spaces.

## Security and limitations

Set `APEX_API_KEY` to require the `X-API-Key` header on decision endpoints. Tool calls remain permission-gated and untrusted tool output is sanitized. ApexDecision is a decision aid; validate outputs before using them in high-impact workflows.

See [docs/api.md](docs/api.md), [docs/tools.md](docs/tools.md), and [SECURITY.md](SECURITY.md).
