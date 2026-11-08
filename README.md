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

---

## Apex-748M Classifier & RLVR Training Suite

ApexDecision also includes the production blueprint and implementation of **Apex-748M** (`apex_classifier_750m/`), a zero-token, non-autoregressive decision model built for enterprise financial and legal verification.

* **Exact Parameter Accounting:** **746,813,455 Parameters** ($d=1536$, 21 Tier-1 layers, 4 Tier-2 cross-attention memory layers).
* **Zero Pretrained Backbone:** Trained 100% from scratch with fresh weights, eliminating all foundation backbone priors, contamination, and quadratic context bottlenecks.
* **No Teacher-Student Distillation:** Completely eliminates teacher LLMs and soft hallucinations in favor of **Reinforcement Learning with Verifiable Rewards (RLVR)** with **GRPO** ($G=4$).
* **Deterministic Sandboxed Math Engine:** Mathematical claims are verified via an AST symbolic oracle enforcing strict magnitude limits ($|x| \le 10^{30}$) and rejecting `bool`, `float('nan')`, and `float('inf')`.
* **512K Chunked Streaming Context:** Employs hierarchical memory latents to deliver an $\approx 942\times$ reduction in attention compute over dense 512K transformers.
* **Kaggle Accelerator Handshake:** Designed for Kaggle's free compute quotas: **20 hours/week on Cloud TPU (v3-8 / v5e-4)** followed seamlessly by **30 hours/week on Dual NVIDIA GPUs (2x T4 / P100)** with portable SHA-256 checkpoints.

For full technical specifications and training instructions:
* **[Apex-748M Architectural Specification (v3.0.0)](docs/APEX_750M_SPECIFICATION.md)**
* **[Kaggle TPU & Dual GPU Execution Guide](docs/KAGGLE_TRAINING_GUIDE.md)**
