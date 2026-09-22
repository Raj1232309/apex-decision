# ApexDecision External Benchmark Suite

A fully independent, reproducible, and non-compensating external evaluation framework for the **ApexStudent-50M** production decision engine.

---

## 1. Overview & Architecture Compliance

ApexDecision V8 evaluates a frozen ~46.53M active parameter student model initialized 100% from scratch with zero pretrained weights and zero external API dependencies at inference time.

In accordance with strict evaluation integrity principles:
- **No benchmark leakage**: Zero hard-coded benchmark answers, templates, or heuristics.
- **Strict evidentiary taxonomy**: Every result is categorized under `DIRECT_HEAD_TO_HEAD`, `LOCAL_REPLICATION`, `LOCAL_STRESS_TEST`, or `DOCUMENTED_PUBLIC`.
- **Interface integrity**: For tasks incompatible with a non-autoregressive typed decision engine, the framework explicitly records `NOT_APPLICABLE` instead of manufacturing a score.
- **Zero-invention verification**: Every factual evidence span is verified by cryptographic provenance matching; fabricated spans score zero.

---

## 2. Benchmark Suite Inventory

The authoritative list of evaluation suites is registered in [`benchmark/manifest.yaml`](manifest.yaml):

| Suite | Benchmark ID | Task Type | Split | Target Metric | Pass Threshold | Source |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Core** | `smoke_production_integrity` | Health & Contract | `dev` | Pass Rate | 100.0% | Local Synthetic |
| **Core** | `typed_decisions_jev_contract` | Wire Contract | `frozen_final` | Schema Compliance | 100.0% | Documented Public |
| **Core** | `grounding_unseen_attribution` | Factual Entailment | `frozen_final` | Accuracy | 95.0% | Independent Synthetic |
| **Core** | `hallucination_refusal` | Refusal Precision | `frozen_final` | Precision | 98.0% | Independent Synthetic |
| **Reasoning** | `reasoning_multihop_arithmetic` | Structured Arithmetic | `frozen_final` | Exact Match | 95.0% | Independent Synthetic |
| **Reasoning** | `reasoning_contradiction_scope` | Logical Consistency | `frozen_final` | Accuracy | 95.0% | Independent Synthetic |
| **Multilingual** | `multilingual_fixed_ood` | 90 Languages / 18 Scripts | `frozen_final` | Accuracy | 98.0% | Independent Synthetic |
| **Tools** | `tools_orchestration_matrix` | 20 Tool Categories | `frozen_final` | Decision Accuracy | 99.0% | Independent Synthetic |
| **Long Context** | `long_context_scaling` | Needle Retrieval (32K-1M) | `frozen_final` | Retrieval Accuracy | 95.0% | Independent Synthetic |
| **External** | `external_eleuther_adapter` | Multiple Choice Adapter | `val` | Accuracy | 80.0% | Public External |
| **External** | `external_opencompass_adapter` | Classification Adapter | `val` | Accuracy / N/A | 80.0% | Public External |
| **Performance** | `performance_latency_profiling` | Latency, VRAM, Throughput | `dev` | P50 Latency | < 25.0 ms | Measured Local |

---

## 3. Quick Start & CLI Usage

### Canonical Execution:
Run the complete external benchmark suite:
```bash
python -m benchmark.run_all
```

### Targeted Execution by Suite:
```bash
# Core smoke, grounding, and typed decision contracts
python -m benchmark.run_all --suite core

# Multi-hop arithmetic DAGs and logical contradictions
python -m benchmark.run_all --suite reasoning

# Multilingual evaluation across 90 world languages
python -m benchmark.run_all --suite multilingual

# Native tool orchestration (Stages A-H across 20 categories)
python -m benchmark.run_all --suite tools

# Hierarchical long context scaling (32K to 1M tokens)
python -m benchmark.run_all --suite long_context

# External LM-Eval and OpenCompass adapters
python -m benchmark.run_all --suite external

# Latency, VRAM, and throughput profiling
python -m benchmark.run_all --suite performance

# Quick CI validation mode
python -m benchmark.run_all --quick
```

---

## 4. Interactive Web Interface (Gradio)

Launch the interactive web-based benchmark center:
```bash
python benchmark/app.py
```
Open your browser at `http://localhost:7860`. The web UI provides:
- Live streaming execution logs
- One-click triggers for all benchmark suites
- Real-time telemetry (GPU VRAM, status, elapsed time)
- Direct downloads of `summary.json`, `metrics.csv`, and `report.html`
- Concurrency lock preventing conflicting concurrent executions

---

## 5. Output Artifacts

Each benchmark run automatically outputs structured artifacts to `results/<timestamp>/`:
- `summary.json`: Top-level summary including aggregate scores, 95% Wilson confidence intervals, and model/hardware metadata.
- `metrics.csv`: Machine-readable tabular breakdown across all evaluated benchmarks.
- `report.html`: Self-contained interactive HTML dashboard.
- `environment.json`: Hardware, PyTorch, CUDA, OS, git commit, and model SHA-256 hashes.
- `benchmark_manifest.json`: Frozen snapshot of `manifest.yaml` for reproducibility.
- `failures.jsonl`: Detailed log of mispredicted cases with automated categorization.
- `raw_logs/`: Timestamped per-suite execution logs.
