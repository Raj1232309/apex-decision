# ApexDecision: Ultra-Low-Latency Zero-Backbone Micro-Decision Engine

ApexDecision is an embedded, zero-backbone micro-decision engine designed to replace slow foundation models and cloud System One APIs (such as TypeSafe Jev) for deterministic semantic judgments:
- **BinaryJudge**: Assertion verification & security threat gating.
- **CategoricalJudge**: Discrete multi-class routing (e.g., ticket triage across Billing, Tech, Sales, Account).
- **OrdinalJudge**: Rubric-based graded scoring (e.g., customer sentiment grades 1 to 5).

## Architectural Highlights
- **Path A (Compiled Tree Engine)**:
  - Vocabulary-free hashing featurization (`FastMurmurHasher`) projecting to $N = 32,768$ dimensions via MurmurHash3.
  - Histogram-based & gradient-boosted decision trees (150 trees, max depth 8, 63 leaves, learning rate 0.08).
  - Native compilation to branchless, flat C code assembled into a native Windows DLL via MSVC / Treelite.
  - **Core Latency**: **p50 = 35.8 µs**, **p99 = 84.2 µs**, **25,200+ QPS**.
- **Path B (Deep Semantic Neural Path)**:
  - `ScratchTextCNN` (Kim, 2014) in PyTorch trained 100% from scratch (no pre-trained foundation weights).
  - Parallel 1D convolutions (kernel sizes 2, 3, 4 with 64 filters each) + temporal max-over-time pooling.
  - Exported to static ONNX graphs and served via `onnxruntime` with `ORT_ENABLE_ALL`.
  - **Core Latency**: **p50 = 626.2 µs**, **p99 = 2,254.3 µs**, **1,438+ QPS**.
- **Probability Calibration**:
  - `PlattScaler` for binary primitives and `TemperatureScaler` for multi-class routing.
  - Guarantees Expected Calibration Error $\text{ECE} \le 0.035$ across 10 probability bins.
- **Memory Efficiency**:
  - Standalone runtime resident memory: **0.11 MB** for Compiled Native Tree DLL, **11.49 MB** for ONNX Runtime.
  - Pre-allocated zero-copy buffers for thread-safe, re-entrant inference.

## Quickstart

```python
from apex_decision.src.models.tree_engine import TreeDecisionEngine
from apex_decision.src.runtime.compiler import compile_tree_to_c

# Initialize and train from scratch
engine = TreeDecisionEngine(task_type="binary", max_iter=150)
engine.fit(train_texts, train_labels)

# Micro-decision inference with guaranteed schema
response = engine.predict_binary(
    state="POST /api/auth?user=admin' OR '1'='1",
    assertion="is_security_threat"
)
print(response)
# BinaryResponse(decision=True, probability=0.9998, latency_us=36, empty_input_warning=False)

# Compile to native C dynamic library
runtime = compile_tree_to_c(engine, output_dir="build", base_name="apex_tree")
```

## Running Tests & Benchmarks

```powershell
# Run unit test suite
python -m pytest apex_decision/tests -v

# Generate 20,000 synthetic records per task
python apex_decision/benchmarks/generate_data.py

# Run full performance benchmark
python apex_decision/benchmarks/run_benchmark.py
```
