# ApexDecision V7.3: Training Efficiency & Bottleneck Audit

**Audit Date:** September 21, 2026  
**Subject System:** `ApexDecision V7.3 Autonomous Two-Hour Self-Improvement Loop`  
**Logged Execution Runs:** 393 Iterations (~5,766 seconds / 96.1 minutes)  
**Target Hardware:** NVIDIA GeForce RTX 2050 (4,096 MiB VRAM), PyTorch 2.6.0+cu124  

---

## 1. Executive Summary: Why Semantic Capability Did Not Move

The V7.3 autonomous training loop successfully demonstrated stable PyTorch AMP training on the RTX 2050 without memory leaks or crashes, executing **393 iterative cycles** across 96.1 minutes. Furthermore, it sharpened uncertainty calibration (ECE improved from **0.1034 to 0.0442**) and trimmed single-query latency (from **5.41 ms to 4.86 ms**).

However, **broad semantic capability remained stagnant**:
- **Grounding Accuracy:** 88.0% $\rightarrow$ 89.0% ($+1.0\%$)
- **Adversarial Accuracy:** 92.41% $\rightarrow$ 93.67% ($+1.26\%$)
- **Capability Stress Score:** **61.29% $\rightarrow$ 61.29% (0.00% change)**
- **Promotion Ratio:** **2 Promoted / 391 Rejected (99.5% Rejection Rate)**

This audit identifies the **three structural flaws** responsible for this plateau:
1. **The "Episodic Reversion Trap" (Lack of Cumulative Gradient Learning):** Checkpoints were reverted back to baseline on 391 out of 393 cycles due to minor evaluation noise on a 120-item validation split. The model never accumulated gradient updates over time; it essentially trained for 20 optimizer steps from baseline, reverted, and repeated this 391 times.
2. **Compute Allocation Inefficiency:** Laya teacher inference was recomputed from scratch on every single cycle without a persistent cache, consuming **44.3% of the total time budget** while actual student training consumed only **25.9%**.
3. **Pure Neural Expectation for Deterministic Arithmetic:** The system forced a 19.5M non-autoregressive encoder to perform mental multi-hop arithmetic and date comparisons without a deterministic fact register or symbolic layer.

---

## 2. Quantitative Time & Compute Breakdown

Analysis of `reports/autonomous_training_log.jsonl` (393 cycles, 5,766.5 total elapsed seconds):

| Phase / Operation | Time per Cycle (Mean) | Total Time Spent | % of Compute Budget | Bottleneck Status |
| :--- | :--- | :--- | :--- | :--- |
| **Laya Teacher Inference** | 6.50 s | 2,554.5 s | **44.3%** | **CRITICAL BOTTLENECK (No Cache)** |
| **Student GPU Training** | 3.80 s | 1,493.4 s | **25.9%** | Under-allocated (~20 steps/cycle) |
| **Validation & Evaluation** | 4.22 s | 1,658.5 s | **28.8%** | Overly frequent Tier-C eval |
| **Synthetic Example Gen** | 0.15 s | 59.0 s | 1.0% | Highly efficient |
| **Checkpoint I/O & Logging** | < 0.01 s | 1.1 s | < 0.1% | Negligible |
| **Total per Cycle** | **14.67 s** | **5,766.5 s** | **100.0%** | Average 4.09 cycles / minute |

### Detailed Throughput & GPU Utilization:
- **GPU Training Batch Size:** 16
- **Training Examples per Cycle:** 326 (163 fresh hard cases + 163 base replay cases)
- **Optimizer Steps per Cycle:** ~20.3 steps
- **Total Optimizer Steps Attempted:** **7,978 steps**
- **Effective Cumulative Steps Retained:** **~40 steps** (due to 99.5% rejection rate reverting weights!)
- **Training Throughput:** **85.8 examples / second** (~32,600 tokens / second during active forward/backward)
- **Peak Discrete GPU VRAM:** 1,847.8 MB (Comfortably under the 4,096 MB hardware ceiling)

---

## 3. Deep Diagnosis of the Failure Modes

### Flaw 1: The Episodic Reversion Trap
In V7.3, the controller logic followed:
```python
if candidate_metrics["fitness_score"] > best_fitness:
    best_fitness = candidate_metrics["fitness_score"]
    torch.save(model.state_dict(), best_ckpt)
else:
    # REVERT BACK TO BASELINE
    model.load_state_dict(torch.load(best_ckpt))
```
Because evaluation was performed on 120 samples, binomial variance ($\sigma \approx \sqrt{p(1-p)/N} \approx 3.2\%$) meant that even a beneficial gradient step had a ~50% probability of scoring 0.5 points lower on a single noisy evaluation batch. When rejected, the entire model was reset to the baseline checkpoint. Consequently, the model was trapped in a 1-epoch oscillation around the initial weights.

### Flaw 2: Redundant Teacher Compute
Every cycle generated 163 synthetic cases and called `laya_agent.predict()` synchronously on CUDA. Across 393 iterations, over **64,000 forward passes** were executed by Laya. Because many prompt structures and sub-clauses share identical premises, lack of a persistent hash cache wasted **42.5 minutes of GPU compute** that should have been spent on student backpropagation.

### Flaw 3: Expecting a 19.5M Model to do Mental Arithmetic in a Single Pass
The capability stress score was frozen at 61.29% because 12 specific tasks repeatedly failed. Among them:
- Numeric near-misses (e.g. 48.3M vs 48.8M) were not caught because `runtime.py` checked `num_conflict` (sign polarity) rather than `not num_consistent`.
- Contradiction rejection verbs (*terminated, rejected, cancelled*) were omitted from `CORE_NEGATORS`.
- Multi-hop additions ($45.0M + $18.5M = $63.5M) were expected to be computed by the neural attention weights rather than a deterministic arithmetic layer.

---

## 4. The V7.4 Architectural Overhaul Mandate

To turn the two-hour compute budget into substantial capability gains:
1. **Three-Tier Evaluation System:**
   - **Tier A (Cheap):** 25 balanced regression items evaluated in <1.0s every cycle.
   - **Tier B (Medium):** 60 capability items evaluated every 5 cycles (~3.0s).
   - **Tier C (Full Frozen Benchmark):** Full 600-sample test suite evaluated only when a candidate achieves sustained gains across Tiers A & B.
2. **Persistent Laya Teacher Cache:**
   - Hash-indexed SQLite/JSON cache storing teacher probabilities and confidences.
   - Eliminates 90%+ of redundant teacher forward passes, freeing ~40 minutes of compute.
3. **Continuous Curriculum Accumulation:**
   - Instead of resetting weights after 20 steps, allow continuous momentum learning over multi-epoch training phases with periodic checkpoint rollback only on certified catastrophic regressions.
4. **Hybrid Neuro-Symbolic Integration (`context/fact_register.py`):**
   - Pair the neural encoder with a deterministic fact register for arithmetic, strict numeric consistency, and temporal ordering.
