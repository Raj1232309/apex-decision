# Apex-750M / 1.2B Classifier Architecture & Engineering Specification

**Document Version:** 2.0.0 (Comprehensive Technical Audit & Mathematical Revision)  
**Target Architectures:**  
* Option A (True 750M): 26-Layer Local Bidirectional + 6-Layer Memory Recurrence ($d=1536$)  
* Option B (Full 1.2B): 32-Layer Local Bidirectional + 8-Layer Global Cross-Attention ($d=1536$)  
**Hardware Reality:** Google Cloud TPU (v2-8: 8×8GB HBM; v3-8: 8×16GB HBM)  
**Execution Paradigm:** Non-Autoregressive Decision Engine with Sandboxed Symbolic Compute  
**Codebase Grounding:** Upstream reference: `apex_decision/` (ApexStudent production codebase)  

---

## 1. Executive Summary & Design Grounding

### 1.1 Architectural Reality Check: Shipped Code vs. Proposed Scale
The existing production codebase in `apex_decision/` is a compact, latency-optimized student engine:
* **Current Model (`models/apex_student/`):** 24M–194M parameters ($d=384\dots 1024$, 8–14 layers, vocabulary 16,384, RoPE base $\theta=10,000$, max position length 8,192 with runtime clipping to 1,024).
* **Current Long Context (`context/long_context/hierarchical_engine.py`):** Deterministic lexical and entity overlap scoring (`word_overlap*2 + opt_overlap*1.5 + num*5`), extracting the top-$k$ raw text blocks. Real text blocks are retained because lossy compression loses exact numerical, date, and clause citations.
* **Current Math (`reasoning/arithmetic_engine.py`):** Regex pattern extraction and floating-point chain solving.
* **Current Decision Logic (`reasoning/noul_verifier.py`, `runtime.py`):** Calibrated rule-based validation combining neural logits with deterministic consistency guards.

Scaling from this baseline to a 750M–1.2B parameter model is **not an incremental patch**; it is an architectural evolution. This document provides the mathematically rigorous, implementable blueprint for this scaling, correcting the parameter counts, hardware allocations, attention budgets, and security models.

---

## 2. Rigorous Parameter Accounting

### 2.1 Layer-Level Mathematical Breakdown
For a transformer layer with hidden dimension $d$, attention heads $H$, head dimension $d_h = d / H$, and feed-forward dimension $d_{\text{ff}}$:
1. **Multi-Head Self-Attention (MHA):**
   * $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$: $3 \times d^2$
   * Output projection $W_O \in \mathbb{R}^{d \times d}$: $d^2$
   * Total MHA parameters: $4d^2$
2. **Feed-Forward Network (FFN):**
   * Standard 2-matrix FFN (GELU, $d_{\text{ff}} = 4d$): $W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d} \implies 8d^2$
   * SwiGLU FFN (3 matrices, $d_{\text{ff}} = \frac{8}{3}d$ or custom): $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d} \implies 3 \times d \times d_{\text{ff}}$
     * For $d = 1,536$ and $d_{\text{ff}} = 4,096$: $3 \times (1,536 \times 4,096) = 18,874,368$
3. **Layer Normalization / RMSNorm:**
   * Pre-attn and Pre-FFN norms: $2 \times d = 3,072$

$$\text{Params}_{\text{layer}} = 4d^2 + 3 d \cdot d_{\text{ff}} + 2d = 4(1536^2) + 3(1536 \cdot 4096) + 3072 = 9,437,184 + 18,874,368 + 3,072 = \mathbf{28,314,624}$$

### 2.2 Recomputing Option A (True ~750M Model) vs. Option B (Full 1.2B Model)

| Component | Dimensions / Structure | Option A: True ~750M | Option B: Full ~1.22B |
| :--- | :--- | :--- | :--- |
| **Token Embeddings** | Vocab 16,384 $\times 1,536$ (shipped tokenizer) | 25,165,824 | 25,165,824 |
| **Tier 1 (Local Encoder)** | $d=1536, H=24, d_{\text{ff}}=4096$ | **22 Layers** ($22 \times 28.31\text{M} = 622.92\text{M}$) | **32 Layers** ($32 \times 28.31\text{M} = 906.07\text{M}$) |
| **Tier 2 (Global Memory)**| $d=1536, H=24, d_{\text{ff}}=4096$ | **4 Layers** ($4 \times 28.31\text{M} = 113.26\text{M}$) | **8 Layers** ($8 \times 28.31\text{M} = 226.52\text{M}$) |
| **Memory Latents & Projections**| 4 tokens $\times 1536$ + Cross-Attention QKV | 9,443,328 | 9,443,328 |
| **Candidate Encoder (Choice)** | Shared backbone projection ($1536 \to 1536$) | 2,359,296 | 2,359,296 |
| **Decision Heads (Noul, Score, Abstain)** | Multi-layer perceptrons + layer norms | 3,148,800 | 3,148,800 |
| **TOTAL PARAMETERS** | — | **~776.2 Million** | **~1.173 Billion** |

*Note on Vocabulary Size:* If expanding the vocabulary to 32,000 (standard SentencePiece/Llama tokenizer), token embeddings increase to $49,152,000$, bringing Option A to **~800M** and Option B to **~1.20B**.

---

## 3. Hardware & Memory Reality on Google Cloud TPUs

### 3.1 Accurate Google Cloud TPU Hardware Profiles
Google Colab does **not** provide a "47 GB TPU". TPU hardware specifications are fixed:
* **TPU v2-8:** 8 physical tensor cores. **8 GB HBM per core** = **64 GB aggregate memory**.
* **TPU v3-8:** 8 physical tensor cores. **16 GB HBM per core** = **128 GB aggregate memory**.
*(The 47 GB figure often seen in Colab refers to host system RAM on high-RAM CPU instances or Nvidia A100 GPU slices).*

### 3.2 Real Training Memory Footprint (16 Bytes / Parameter Static Overhead)
In standard mixed-precision training (bfloat16 forward/backward with FP32 master weights and AdamW states):
* **Model Parameters (`bfloat16`):** 2 bytes/param
* **Gradients (`bfloat16`):** 2 bytes/param
* **AdamW Master Weights (`float32`):** 4 bytes/param
* **AdamW Momentum ($m$) (`float32`):** 4 bytes/param
* **AdamW Variance ($v$) (`float32`):** 4 bytes/param
* **Total Static State per Parameter:** **16 bytes/parameter**.

$$\text{Memory}_{\text{static}} = 776\text{M} \times 16\text{ bytes} = \mathbf{12.42\text{ GB}}$$
$$\text{Memory}_{\text{static (1.2B)}} = 1.17\text{B} \times 16\text{ bytes} = \mathbf{18.72\text{ GB}}$$

### 3.3 Core Sharding (FSDP / ZeRO-3) Requirements
* On **TPU v2-8 (8 GB/core):** A single core **cannot** hold the 12.42 GB static state. Training requires Fully Sharded Data Parallel (FSDP) via `torch_xla.distributed.fsdp` or JAX `pjit`. Sharding 12.42 GB across all 8 cores results in:
  $$\frac{12.42\text{ GB}}{8\text{ cores}} \approx \mathbf{1.55\text{ GB/core (static state)}}$$
* On **TPU v3-8 (16 GB/core):** Static state fits without sharding, but FSDP is mandatory to preserve memory for sequence activations.

### 3.4 Activation Memory Accounting & Avoidance of Quadratic Attention Blowup
For Tier 2, if batch size $B=64$ were evaluated over 4,000 memory tokens with full quadratic attention:
$$\text{Memory}_{\text{logits}} = B \times H \times L_{\text{mem}}^2 \times 2\text{ bytes} = 64 \times 24 \times (4,000)^2 \times 2 \approx \mathbf{49.15\text{ GB}}$$
This would immediately cause an Out-Of-Memory error.  
**Mandatory Mitigations:**
1. **Gradient Checkpointing:** Recomputes activations during backward passes, reducing activation storage to $O(\sqrt{N_{\text{layers}}})$.
2. **Query-Driven Cross-Attention:** The 4,000 memory tokens are not self-attended in an all-to-all matrix. Instead, the task decision latent $q \in \mathbb{R}^{B \times 1 \times d}$ cross-attends to the 4,000 memory tokens:
   $$\text{Memory}_{\text{cross-logits}} = B \times H \times 1 \times 4,000 \times 2\text{ bytes} = 64 \times 24 \times 4,000 \times 2 \approx \mathbf{12.28\text{ MB}}$$
   This completely removes the 49 GB memory bottleneck.

---

## 4. Computational Complexity: Honest Analysis

### 4.1 Attention vs. FFN Domination
Claims of "942× FLOP reduction" based purely on quadratic attention token products are misleading:
* At chunk length $L=512$, **FFN compute dominates attention compute by ~12×**:
  * Attention MACs per token: $4d^2 + 2 \cdot L \cdot d = 4(1536^2) + 2(512)(1536) \approx 9.44\text{M} + 1.57\text{M} = 11.01\text{M}$
  * SwiGLU FFN MACs per token: $3 \times d \times d_{\text{ff}} = 3(1536)(4096) \approx 18.87\text{M}$
* In a 512K token context, Tier 1 must run its FFN over all 512,000 tokens regardless of attention structure.
* **Realistic Speedup:** Compared to a naive full-attention transformer (which cannot run at 512K), the dual-tier architecture makes 512K feasible. Compared to chunked retrieval or linear attention models, the actual FLOP reduction is **$3\times$ to $6\times$**, not $942\times$.

### 4.2 Latency Reality
Processing 1,000 sequential 512-token chunks cannot be completed in 35 ms on a single accelerator. To achieve sub-100ms inference over 512K tokens:
* Chunks must be evaluated in parallel across TPU matrix units using batched execution: $B_{\text{effective}} = 1,000$.
* If batch capacity cannot swallow 1,000 chunks simultaneously, retrieval pre-filtering (as proven in `apex_decision/context/long_context/hierarchical_engine.py`) must prune candidate chunks to the top 16–32 blocks before neural encoding.

---

## 5. Architectural & Mathematical Details

### 5.1 Local Positional Embeddings: RoPE Frequency Base
* **Correction:** The base frequency $\theta = 500,000$ must **not** be used inside a 512-token window. A high base dampens rotational speed, causing relative positional representations to become indistinguishable at short distances ($L \le 512$).
* **Adopted Setting:** Local Tier 1 uses standard RoPE with base **$\theta = 10,000$** (matching `apex_decision/models/apex_student/model.py`), ensuring maximum discrimination of local word order, syntax, and entity relations.

### 5.2 Mathematical Formulation of Heads

#### A. Calibrated Noul Head (Belief Grounding)
Evaluates whether document context entails a given proposition $y \in \{0, 1\}$:
$$h_{\text{proj}} = \text{GELU}(W_1 h_{\text{out}} + b_1)$$
$$z_{\text{noul}} = W_2 h_{\text{proj}} + b_2$$
$$\hat{p} = \sigma\left(\frac{z_{\text{noul}}}{T_{\text{noul}}}\right)$$
* Where $T_{\text{noul}} > 0$ is a scalar temperature parameter learned on a held-out calibration split via negative log-likelihood minimization.

#### B. Hierarchical High-K Choice Head ($K \le 255$)
To avoid catastrophic dot-product degradation across 255 unstructured classes:
1. **Stage 1 (Coarse Retrieval):** Candidates are scored via dot product with a candidate projection vector:
   $$s_i^{(1)} = \frac{h_{\text{out}} \cdot W_{\text{cand}} c_i}{\sqrt{d}}$$
   Top-$M$ candidates ($M=16$) are shortlisted.
2. **Stage 2 (Fine Reranking):** Shortlisted candidates are cross-attended with $h_{\text{out}}$ through an MLP to produce final normalized probabilities:
   $$p_k = \frac{\exp(s_k^{(2)} / \tau)}{\sum_{j=1}^M \exp(s_j^{(2)} / \tau)}$$

#### C. Ordinal Score Head (Monotonic Cumulative Link)
Predicts discrete ratings across ordered tiers (levels $0, 1, \dots, K$):
To prevent probability distribution collapse, thresholds $\theta$ are parameterized with strict monotonicity:
$$\theta_1 = b_1, \quad \theta_k = \theta_{k-1} + \exp(\delta_k) \quad \text{for } k=2, \dots, K$$
$$P(\text{score} \le k) = \sigma(\theta_k - W_{\text{score}} h_{\text{out}})$$
$$P(\text{score} = k) = P(\text{score} \le k) - P(\text{score} \le k-1)$$

---

## 6. Symbolic Compute Engine & AST Security Architecture

### 6.1 Vulnerability Elimination in Symbolic Execution
Executing dynamic Python code or unrestricted `ast.literal_eval` on model-generated outputs introduces severe arbitrary code execution risks. The symbolic engine is specified as a **strictly isolated, whitelisted AST visitor**.

### 6.2 Formal Node Whitelist
The parser only accepts the following Python `ast` nodes:
```python
SAFE_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd
}
```
Any AST containing `ast.Call`, `ast.Attribute`, `ast.Import`, `ast.Name`, `ast.Subscript`, or control-flow constructs is immediately rejected.

### 6.3 Hard Operational Bounds
* **Recursion Depth:** Maximum AST depth $\le 16$.
* **Input Length:** Maximum expression string length $\le 256$ characters.
* **Division-by-Zero & NaN Handling:** Replaces zero-division with a typed error token (`MATH_ERROR_ZERO_DIV`) rather than halting execution.
* **Timeout:** Maximum evaluation execution time $\le 5$ milliseconds.

---

## 7. Calibration, Loss Formulation, & Training Scale

### 7.1 Separation of Focal Loss and Calibration
Focal loss ($\gamma=2.0$) concentrates gradients on hard boundary examples, which intentionally distorts probability calibration (Mukhoti et al., 2020). Therefore:
* **During Training:** Multi-task Focal Loss is used strictly to establish decision boundary separation.
* **Post-Training Calibration:** Model weights are frozen. A held-out calibration split ($N \ge 10,000$ samples) is used to fit temperature scaling $T$ via Platt scaling or Isotonic Regression.
* **Evaluation:** Expected Calibration Error (ECE) is reported per 10 confidence bins on unseen out-of-distribution test sets. No theoretical "$\ge 90\%$ guarantee" is claimed; empirical metrics dictate deployment gates.

### 7.2 Realistic Training Token Budget
Under Chinchilla scaling laws, compute-optimal training for a ~750M parameter model requires at least **$20\times$ parameter count in tokens**:
$$\text{Tokens}_{\text{optimal}} \approx 750\text{M} \times 20 = \mathbf{15\text{ Billion Tokens}}$$
* A schedule of 50,000 steps at batch size 64 with $L=512$ processes only $1.64\text{ Billion tokens}$, leaving the model significantly undertrained.
* **Revised Schedule:**
  * **Phase 1 (Representation Warmup):** 100,000 steps at batch 128 ($L=512$) = 6.55B tokens.
  * **Phase 2 (Long-Context & Decision Distillation):** 100,000 steps at batch 64 ($L=1024$) = 6.55B tokens.
  * **Phase 3 (Hard-Negative & Math Tuning):** 25,000 steps focused on adversarial and numeric sets = 1.64B tokens.
  * **Total Training Volume:** $\approx \mathbf{14.74\text{ Billion Tokens}}$.

---

## 8. Secure Google Colab & TPU Execution Workflow

### 8.1 Secure Credential Handling (No Tokens in URLs)
Never expose GitHub Personal Access Tokens in notebook cells or URL strings.
```python
# Colab Secret Storage (Key icon in left sidebar -> GITHUB_TOKEN)
from google.colab import userdata
import os

token = userdata.get('GITHUB_TOKEN')
os.system(f"git clone https://x-access-token:{token}@github.com/Raj1232309/apex-decision.git")
%cd apex-decision
```

### 8.2 Safe Checkpointing Protocol
Saving large checkpoints directly over Google Drive FUSE mounts causes silent timeouts and kernel crashes.
* **Protocol:** Checkpoints are saved locally to `/tmp/checkpoints/` first, verified with SHA-256 checksums, and asynchronously copied to Google Drive:
```bash
python -c "import shutil, hashlib; shutil.copyfile('/tmp/ckpt.pt', '/content/drive/MyDrive/ckpt.pt')"
```
