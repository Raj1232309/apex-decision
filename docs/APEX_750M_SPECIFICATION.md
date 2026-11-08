# Apex-748M / 776M Classifier Architecture & Engineering Specification

**Document Version:** 3.0.0 (Production-Audited Specification: RLVR Paradigm)  
**Target Architectures:**  
* **Apex-748M (Exact 750M-Class Target):** 21-Layer Local Bidirectional + 4-Layer Memory Recurrence ($d=1536$) — **746,813,455 Parameters**  
* **Apex-776M (Expanded 26-Layer Variant):** 22-Layer Local Bidirectional + 4-Layer Memory Recurrence ($d=1536$) — **775,128,079 Parameters**  
* **Apex-1.17B (Full Reference Configuration):** 32-Layer Local Bidirectional + 8-Layer Global Cross-Attention ($d=1536$) — **1,171,532,815 Parameters**  
**Hardware Target:** Google Cloud TPU (v2-8: 8×8 GB = 64 GB HBM; v3-8: 8×16 GB = 128 GB HBM; TPU VM or Colab Pro)  
**Execution Paradigm:** Non-Autoregressive Zero-Token Decision Engine with Sandboxed AST Symbolic Engine & Reinforcement Learning with Verifiable Rewards (RLVR)  
**Codebase Grounding:** Upstream reference: `apex_decision/` (ApexStudent production codebase)  

---

## 1. Executive Summary & Design Grounding

### 1.1 Architectural Reality Check: Shipped Code vs. Proposed Scale
The existing production codebase in `apex_decision/` is a compact, latency-optimized student engine:
* **Current Model (`models/apex_student/`):** 24M–194M parameters ($d=384\dots 1024$, 8–14 layers, vocabulary 16,384, RoPE base $\theta=10,000$, max position length 8,192 with runtime clipping to 1,024).
* **Current Long Context (`context/long_context/hierarchical_engine.py`):** Deterministic lexical and entity overlap scoring (`word_overlap*2 + opt_overlap*1.5 + num*5`), extracting the top-$k$ raw text blocks. Real text blocks are retained because lossy compression loses exact numerical, date, and clause citations.
* **Current Math (`reasoning/arithmetic_engine.py`):** Regex pattern extraction and floating-point chain solving.
* **Current Decision Logic (`reasoning/noul_verifier.py`, `runtime.py`):** Rule-based validation combining neural logits with deterministic consistency guards.

Scaling from this baseline to a 750M-class model is an architectural evolution. This document establishes the exact, audited blueprint for this scaling, with mathematically verified parameter counts, real hardware allocations, an enforceable AST sandbox, and realistic training economics.

---

## 2. Rigorous Parameter Accounting

### 2.1 Layer-Level Mathematical Breakdown
For a transformer layer with hidden dimension $d$, attention heads $H$, head dimension $d_h = d / H$, and SwiGLU feed-forward dimension $d_{\text{ff}}$:
1. **Multi-Head Self-Attention (MHA):**
   * $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$: $3 \times d^2 = 3 \times 1536^2 = 7,077,888$
   * Output projection $W_O \in \mathbb{R}^{d \times d}$: $d^2 = 1536^2 = 2,359,296$
   * Total MHA parameters per layer: $4d^2 = \mathbf{9,437,184}$
2. **SwiGLU Feed-Forward Network (FFN):**
   * 3 matrices ($W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$):
   * For $d = 1,536$ and $d_{\text{ff}} = 4,096$: $3 \times (1,536 \times 4,096) = \mathbf{18,874,368}$
3. **Layer Normalization / RMSNorm:**
   * Pre-attn and Pre-FFN norms: $2 \times d = 2 \times 1,536 = \mathbf{3,072}$

$$\text{Params}_{\text{layer}} = 4d^2 + 3 d \cdot d_{\text{ff}} + 2d = 9,437,184 + 18,874,368 + 3,072 = \mathbf{28,314,624}$$

### 2.2 Decision Head Parameter Breakdown (1,979,407 Parameters)
The decision heads are shared identically across all configurations:
* **Noul Head (Binary Entailment / Proposition Verification):**
  * LayerNorm: $2 \times 1,536 = 3,072$
  * Linear ($1,536 \to 768$): $1,536 \times 768\text{ weights} + 768\text{ bias} = 1,180,416$
  * Linear ($768 \to 1$): $768 \times 1\text{ weights} + 1\text{ bias} = 769$
  * Scalar temperature $T_{\text{noul}}$: $1$
  * **Noul Subtotal:** $3,072 + 1,180,416 + 769 + 1 = \mathbf{1,184,258}$
* **Score Head (Ordinal Cumulative Link for 11 tiers: $0 \dots 10$, $K=10$):**
  * LayerNorm: $2 \times 1,536 = 3,072$
  * Linear ($1,536 \to 512$): $1,536 \times 512\text{ weights} + 512\text{ bias} = 786,944$
  * Linear ($512 \to 1$ latent utility $u$): $512 \times 1\text{ weights} + 1\text{ bias} = 513$
  * 10 Cutpoint Thresholds ($\theta_0 = b_0$ and 9 positive increments $\delta_1 \dots \delta_9$ for tiers $0 \dots 10$): $10$
  * **Score Subtotal:** $3,072 + 786,944 + 513 + 10 = \mathbf{790,539}$
* **Choice Head (High-Cardinality Candidate Matching, $K \le 255$):**
  * Candidates are encoded via candidate projection $W_{\text{cand}} \in \mathbb{R}^{1536 \times 1536}$ ($2,359,296$ parameters, budgeted in §2.3 Candidate Projection row).
  * Candidate scoring operates via single-pass dot product $s_i = \frac{h_{\text{out}} \cdot c_i}{\tau}$, with no redundant intermediate MLP.
  * Head parameters: Query LayerNorm ($3,072$) + scalar temperature $\tau$ ($1$) = **3,073**
* **Abstention Head (Epistemic Uncertainty):**
  * Linear ($1,536 \to 1$): $1,536\text{ weights} + 1\text{ bias} = \mathbf{1,537}$
* **Total Decision Heads:** $1,184,258 + 790,539 + 3,073 + 1,537 = \mathbf{1,979,407}$ parameters.

### 2.3 Audited Model Configurations

| Component | Dimensions / Structure | Apex-748M (Exact 750M Target) | Apex-776M (26-Layer Variant) | Apex-1.17B (Full Reference) |
| :--- | :--- | :--- | :--- | :--- |
| **Token Embeddings** | Vocab 16,384 $\times 1,536$ | 25,165,824 | 25,165,824 | 25,165,824 |
| **Tier 1 (Local Encoder)** | $d=1536, H=24, d_{\text{ff}}=4096$ | **21 Layers** ($594,607,104$) | **22 Layers** ($622,921,728$) | **32 Layers** ($906,067,968$) |
| **Tier 2 (Global Memory)**| $d=1536, H=24, d_{\text{ff}}=4096$ | **4 Layers** ($113,258,496$) | **4 Layers** ($113,258,496$) | **8 Layers** ($226,516,992$) |
| **Memory Latents & Projections**| 4 latents $\times 1536$ + Cross-Attn Projections | 9,443,328 | 9,443,328 | 9,443,328 |
| **Candidate Projection (Choice)**| Linear projection $W_{\text{cand}} \in \mathbb{R}^{1536 \times 1536}$ | 2,359,296 | 2,359,296 | 2,359,296 |
| **Decision Heads (Noul, Score, Choice, Abstain)**| Shared MLPs + LayerNorms + Scalar Parameters | 1,979,407 | 1,979,407 | 1,979,407 |
| **EXACT SUM OF PARAMETERS** | — | **746,813,455 (~747M)** | **775,128,079 (~775M)** | **1,171,532,815 (~1.172B)** |

*Vocabulary Expansion Note:* If vocabulary is expanded from 16,384 to 32,000, embeddings become $49,152,000$ params ($+23.99\text{M}$ params), resulting in **~771M** for the 25-layer configuration and **~799M** for the 26-layer configuration.

---

## 3. Hardware & Memory Reality on Google Cloud TPUs

### 3.1 True Hardware Specifications
Google Colab does **not** provide a "47 GB TPU". TPU hardware specifications are fixed:
* **TPU v2-8:** 8 physical tensor cores. **8 GB HBM per core** = **64 GB aggregate memory**.
* **TPU v3-8:** 8 physical tensor cores. **16 GB HBM per core** = **128 GB aggregate memory**.

### 3.2 Real Training Memory Footprint (16–18 Bytes / Parameter Static Overhead)
In mixed-precision training (bfloat16 forward/backward with FP32 master weights and AdamW states):
* **Model Parameters (`bfloat16`):** 2 bytes/param
* **Gradients (`bfloat16` or `float32`):** 2 bytes/param (BF16 accumulation lower bound) to 4 bytes/param (standard PyTorch-XLA autograd FP32 accumulation baseline)
* **AdamW Master Weights (`float32`):** 4 bytes/param
* **AdamW Momentum ($m$) (`float32`):** 4 bytes/param
* **AdamW Variance ($v$) (`float32`):** 4 bytes/param
* **Total Static State per Parameter:** **16 bytes/param (lower bound)** to **18 bytes/param (PyTorch-XLA baseline)**.

$$\text{Memory}_{\text{static (748M, 16B/param lower bound)}} = 746,813,455 \times 16\text{ bytes} = \mathbf{11.95\text{ GB}}$$
$$\text{Memory}_{\text{static (748M, 18B/param PyTorch-XLA baseline)}} = 746,813,455 \times 18\text{ bytes} = \mathbf{13.44\text{ GB}}$$
$$\text{Memory}_{\text{static (776M, 18B/param PyTorch-XLA baseline)}} = 775,128,079 \times 18\text{ bytes} = \mathbf{13.95\text{ GB}}$$

### 3.3 Sharding & Parallelism Strategy
* On **TPU v2-8 (8 GB/core):** A single core **cannot** hold the 12.0–13.4 GB static state. Training requires PyTorch's native Fully Sharded Data Parallel (FSDP):
  ```python
  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
  # Shards static state across all 8 XLA cores:
  # 13.4 GB / 8 cores = ~1.68 GB static memory per core
  ```
* On **TPU v3-8 (16 GB/core):** Static state (13.4 GB) fits on a single core. Standard Data Parallelism (DP) with micro-batching is viable, though FSDP is strongly recommended to free HBM for activation buffers during long-sequence batches.

### 3.4 Activation Memory Accounting
* **Linear Memory Scaling:** In Tier 2, the decision latent $q \in \mathbb{R}^{B \times 1 \times d}$ cross-attends to the 4,000 memory tokens. The attention score matrix is $B \times H \times 1 \times 4000 \times 2\text{ bytes} = \mathbf{12.28\text{ MB}}$ for batch size $B=64$, eliminating quadratic attention matrix scaling.
* **Linear Projection & FFN Buffers:** Storing Key/Value projections and FFN activations for 4,000 tokens across batch 64 requires $64 \times 4000 \times 1536 \times 2\text{ bytes} \approx \mathbf{786\text{ MB}}$ per buffer.
* **Boundary-Only Activation Checkpointing:** Activations are stored strictly at transformer layer boundaries (rather than caching all intra-layer intermediate projection and GELU tensors), recomputing intermediate activations on-the-fly during the backward pass.

---

## 4. Computational Complexity: Honest Analysis

### 4.1 Attention vs. FFN FLOP Distribution
At sequence length $L=512$, SwiGLU FFN operations dominate total layer compute:
* SwiGLU FFN MACs per token: $3 \times d \times d_{\text{ff}} = 3(1536)(4096) \approx \mathbf{18.87\text{M}}$
* Total Attention MACs per token: $4d^2 + 2 \cdot L \cdot d = 4(1536^2) + 2(512)(1536) \approx 9.44\text{M} + 1.57\text{M} = \mathbf{11.01\text{M}}$
* Ratio of FFN to total attention: $\frac{18.87\text{M}}{11.01\text{M}} \approx \mathbf{1.71\times}$.
* *(The $\approx 12\times$ factor applies specifically when comparing FFN compute ($18.87\text{M}$) strictly against the sequence-dependent core attention term $2 \cdot L \cdot d = 1.57\text{M}$, excluding projection matrices).*

#### 512K FLOP Reduction Derivation
* Dense full-length quadratic attention over 512,000 tokens requires $2 \cdot L^2 \cdot d = 2(512000^2)(1536) \approx 8.05 \times 10^{14}$ attention FLOPs.
* Apex-748M chunked hierarchical attention processes 1,000 chunks of 512 tokens ($1000 \times 2(512^2)(1536) \approx 8.05 \times 10^{11}$ FLOPs) plus Tier-2 cross-attention over 4,000 memory latents ($2(4000^2)(1536) \approx 4.92 \times 10^{10}$ FLOPs), totaling $8.54 \times 10^{11}$ attention FLOPs — an **$\approx 942\times$ reduction in sequence-dependent attention compute**.
* When taking into account total layer compute (including the parameter-dense SwiGLU FFN which evaluates all tokens), and assuming the identical 25-layer configuration on both sides, overall compute decreases from $\approx 2.45 \times 10^{15}$ FLOPs down to $\approx 3.95 \times 10^{13}$ FLOPs across the model, delivering an **$\approx 62\times$ end-to-end compute reduction** compared to dense 512K quadratic transformers.

### 4.2 Dense 512K Execution Math: Streaming Accumulation Path
To evaluate 512K tokens without OOM or retrieval pruning:
* **The Architecture Path:** **Streaming Chunked Accumulation**.
* 512,000 tokens = 1,000 chunks of 512 tokens.
* Running all 1,000 chunks in parallel on a single core would require $512,000$ active token activations, exceeding core memory.

#### Multi-Core Pod Sharding (8 Cores)
* When distributed across all 8 cores of a TPU v3-8 pod:
  * Each core processes 32 chunks per step: $8 \times 32 = \mathbf{256\text{ chunks per round}}$.
  * To evaluate all 1,000 chunks requires $\lceil 1000 / 256 \rceil = \mathbf{4\text{ rounds}}$ across the pod.
  * In a single-core environment, 1,000 chunks evaluated in batches of 32 requires $\lceil 1000 / 32 \rceil = \mathbf{32\text{ rounds}}$.

#### Rigorous Compute-Bound Latency Accounting
Evaluating a 512,000-token context through a 746.8M parameter model is strictly compute-bound:
* Total forward FLOPs: $512,000\text{ tokens} \times 2 \times 746.8 \times 10^6 \approx \mathbf{764.7\text{ TFLOPs}}$.
* Per-round compute on 8 cores ($16,384\text{ tokens/core}$): $16,384 \times 2 \times 746.8 \times 10^6 \approx 24.47\text{ TFLOPs/core}$.
* At sustained hardware throughput ($18.375\text{ TFLOPs/core}$ sustained on TPU v3-8 at 35% MFU):
  * **Per-round wall-clock execution:** $24.47 / 18.375 \approx \mathbf{1.33\text{ seconds/round}}$.
  * **Full dense 512K document evaluation (4 rounds):** $4 \times 1.33\text{s} \approx \mathbf{5.2\text{ - }5.3\text{ seconds}}$ on a TPU v3-8 pod ($\mathbf{\approx 1.8\text{ seconds}}$ at theoretical 100% peak 420 TFLOPs).
  * **On a TPU v3-32 pod slice ($588\text{ TFLOPs}$ sustained):** $764.7 / 588 \approx \mathbf{1.30\text{ seconds}}$ per full 512K document pass.
* **Interactive Local Chunk / Short Query Latency:**
  * For local chunk reasoning or short queries ($512 - 2,048$ tokens), compute is only $0.76 - 3.06\text{ TFLOPs}$, yielding an interactive response latency of **$\mathbf{\approx 5.2\text{ ms} - 25\text{ ms}}$**.

---

## 5. Architectural & Mathematical Details

### 5.1 Local Positional Embeddings: RoPE Frequency Base
* **Local Window Setting:** Tier 1 uses Rotary Position Embeddings (RoPE) with base **$\theta = 10,000$** (matching `apex_decision/models/apex_student/model.py`). This guarantees high positional resolution and syntactic sensitivity across the local 512-token chunk.
* High bases ($\theta \ge 500,000$) are restricted to global contexts where sequence lengths span hundreds of thousands of tokens without chunking.

### 5.2 Mathematical Formulation of Heads

#### A. Calibrated Noul Head (Belief Grounding)
Evaluates whether document context entails a given proposition $y \in \{0, 1\}$:
$$h_{\text{proj}} = \text{GELU}(W_1 h_{\text{out}} + b_1)$$
$$z_{\text{noul}} = W_2 h_{\text{proj}} + b_2$$
$$\hat{p} = \sigma\left(\frac{z_{\text{noul}}}{T_{\text{noul}}}\right), \quad \text{where } T_{\text{noul}} = \text{softplus}(\alpha_{\text{noul}}) + \epsilon \quad (\alpha_{\text{noul}} \in \mathbb{R}, \epsilon = 10^{-4})$$
* Where parameterizing $T_{\text{noul}}$ via softplus strictly enforces positivity ($T_{\text{noul}} > 0$) without artificial clamping, preventing division-by-zero during optimization. Scalar parameter $\alpha_{\text{noul}}$ is fitted on a held-out calibration split via negative log-likelihood minimization.

#### B. Unified High-K Choice Head ($K \le 255$)
To ensure calibration across varying candidate set sizes without subset distortion:
* **Candidate Representation ($c_i$):** Each candidate option $i \in \{1, \dots, K\}$ has a text string $x_i^{\text{cand}}$. Its tokens are passed through the shared token embedding layer, mean-pooled to vector $e_i \in \mathbb{R}^d$, and linearly projected:
  $$c_i = W_{\text{cand}} e_i \in \mathbb{R}^d, \quad W_{\text{cand}} \in \mathbb{R}^{d \times d} \quad (2,359,296\text{ params})$$
* **Single-Scaled Logits:** Logits are scaled by temperature $\tau_{\text{choice}}$ (initialized to $\sqrt{d}$):
  $$s_i = \frac{h_{\text{out}} \cdot c_i}{\tau_{\text{choice}}}$$
* **Probability Distribution:** Normalized across all $K$ candidates via full softmax:
  $$p_i = \frac{\exp(s_i)}{\sum_{j=1}^K \exp(s_j)}$$
  This maintains calibrated, comparable probabilities for any candidate count $K \in [2, 255]$.

#### C. Ordinal Score Head (Monotonic Cumulative Link)
Predicts discrete ratings across ordered tiers $k \in \{0, 1, \dots, K\}$ (e.g., 11 levels for $K=10$):
Thresholds $\theta_0 < \theta_1 < \dots < \theta_{K-1}$ are parameterized with strict positive increments:
$$\theta_0 = b_0, \quad \theta_k = \theta_{k-1} + \exp(\delta_k) \quad \text{for } k=1, \dots, K-1$$
*(For an 11-level rating $0 \dots 10$, $K=10$, there are exactly $K=10$ cutpoint thresholds $\theta_0 \dots \theta_9$ delineating the intervals).*

The latent continuous utility $u$ is projected through a 2-layer MLP:
$$u = W_2 \cdot \text{GELU}(W_1 h_{\text{out}} + b_1) + b_2 \in \mathbb{R}$$
Cumulative probabilities are defined as:
$$C_k = P(\text{score} \le k) = \sigma(\theta_k - u) \quad \text{for } k \in \{0, \dots, K-1\}$$
The valid, normalized probability mass function is guaranteed by setting:
$$P(\text{score} = 0) = C_0$$
$$P(\text{score} = k) = C_k - C_{k-1} \quad \text{for } 1 \le k \le K-1$$
$$P(\text{score} = K) = 1.0 - C_{K-1}$$
$$\sum_{k=0}^K P(\text{score} = k) = C_0 + \sum_{k=1}^{K-1} (C_k - C_{k-1}) + (1.0 - C_{K-1}) \equiv 1.0$$

*Head Activation Note (GELU vs. Backbone SwiGLU):*
While the transformer backbone employs SwiGLU for maximum capacity in wide FFN layers ($d_{\text{ff}} = 4096$), the decision heads project down to a scalar or continuous utility ($1536 \to 768 \to 1$ and $1536 \to 512 \to 1$). Standard GELU is chosen for the heads to avoid the $3\times$ matrix projection overhead of gated units in bottleneck layers, maintaining minimal parameter count and matching the production baseline in `apex_decision/models/apex_student/model.py`.

---

## 6. Sandboxed Symbolic Compute Engine

### 6.1 Strict AST Security Specification
To eliminate arbitrary code execution and DoS vulnerabilities, mathematical expressions are evaluated through a sandboxed AST visitor:

```python
import ast
import math
import decimal
from decimal import Decimal, Context, Overflow, DivisionByZero, InvalidOperation

# Strict financial precision context with magnitude bounding
MATH_CONTEXT = Context(prec=28, Emax=999, Emin=-999, traps=[Overflow, DivisionByZero, InvalidOperation])

SAFE_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.USub, ast.UAdd
}

class SecurityError(Exception):
    """Raised when an expression attempts unsafe operations."""
    pass

class SandboxedMathEvaluator(ast.NodeVisitor):
    def __init__(self, max_steps: int = 100):
        self.steps = 0
        self.max_steps = max_steps

    def evaluate(self, expr_str: str) -> Decimal:
        """Parses and evaluates expression within a single context activation."""
        self.steps = 0
        parsed = ast.parse(expr_str, mode='eval')
        with decimal.localcontext(MATH_CONTEXT):
            return self.visit(parsed)

    def visit(self, node):
        self.steps += 1
        if self.steps > self.max_steps:
            raise ValueError("MATH_EVAL_STEP_LIMIT_EXCEEDED")
        if type(node) not in SAFE_NODES:
            raise SecurityError(f"DISALLOWED_NODE: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        # 1. Reject bool (bool is a subclass of int in Python)
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise TypeError("DISALLOWED_CONSTANT_TYPE")
        # 2. Reject oversized literals first (prevents float overflow in math.isfinite for huge ints)
        if abs(node.value) > 1e15:
            raise OverflowError("CONSTANT_MAGNITUDE_EXCEEDED")
        # 3. Reject NaN and Infinities (safe to call since abs(value) <= 1e15)
        if not math.isfinite(node.value):
            raise ValueError("NON_FINITE_NUMERIC_CONSTANT")
        return Decimal(str(node.value))

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise SecurityError(f"DISALLOWED_UNARY_OP: {type(node.op).__name__}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        
        try:
            if op_type is ast.Add: result = left + right
            elif op_type is ast.Sub: result = left - right
            elif op_type is ast.Mult: result = left * right
            elif op_type is ast.Div:
                if right == 0: raise ZeroDivisionError("MATH_ZERO_DIV")
                result = left / right
            elif op_type is ast.FloorDiv:
                if right == 0: raise ZeroDivisionError("MATH_ZERO_DIV")
                result = (left // right)
            elif op_type is ast.Mod:
                if right == 0: raise ZeroDivisionError("MATH_ZERO_DIV")
                result = left % right
            else:
                raise SecurityError(f"OPERATOR_NOT_PERMITTED: {op_type.__name__}")
                
            if abs(result) > Decimal('1e30'):
                raise OverflowError("INTERMEDIATE_MAGNITUDE_EXCEEDED")
            return result
        except (Overflow, DivisionByZero, InvalidOperation) as e:
            raise ArithmeticError(f"ARITHMETIC_FAULT: {str(e)}")
```

### 6.2 Hard Security Constraints & Arithmetic Semantics
* **`ast.Pow` Omission:** Exponentiation is omitted to prevent algorithmic complexity attacks ($9^{9^9}$).
* **Type & Value Guards:** Explicit rejection of `bool`, `float('nan')`, `float('inf')`, and `-inf`.
* **UnaryOp Handler:** Full support for `-x` and `+x` without runtime crashes.
* **Intermediate Magnitude Bound:** Bounded by $|x| \le 10^{30}$ after every operation.
* **Step Limit:** Maximum $\le 100$ AST node visits enforced in Python, preventing nested expression hangs.
* **Division & Modulo Semantics Note (Financial Math):**
  In native Python integer arithmetic, `//` floors towards negative infinity (e.g., `-7 // 3 == -3` and `-7 % 3 == 2`). In `decimal.Decimal` (IEEE-754 standard arithmetic), `//` truncates towards zero (e.g., `Decimal('-7') // Decimal('3') == Decimal('-2')` and `Decimal('-7') % Decimal('3') == Decimal('-1')`). For financial statements and accounting allocations (truncating integer dollar portions towards zero), Decimal's truncated division is standard; however, financial verification harnesses comparing against native Python integer math must account for negative operand differences.

---

## 7. Training Paradigm: Reinforcement Learning with Verifiable Rewards (RLVR)

### 7.1 Motivation: Why RLVR Replaces Teacher-Student Distillation
Teacher-student distillation forces a student model to mimic the soft logit distribution of a large autoregressive generative model. However, frontier generative LLMs suffer from well-documented systemic vulnerabilities:
1. **Arithmetic Hallucinations:** Large LLMs frequently make subtle mathematical errors on multi-step financial calculations. Distillation forces the student to memorize these soft hallucinations.
2. **Sycophancy & Spurious Correlates:** Generative teachers often rely on superficial text formatting, polite phrasing, and keyword density rather than causal syntactic evidence.
3. **No Direct Supervision from Code/Logic:** A generative teacher produces text tokens, not exact categorical distributions.

**Reinforcement Learning with Verifiable Rewards (RLVR)** completely eliminates the teacher model from the training loop. Instead, the model acts as a policy evaluated directly against **deterministic, provable ground truth**: the Sandboxed AST Math Engine, formal logic constraints, and counterfactual invariant checks.

```mermaid
flowchart TD
    subgraph Env ["Verifiable Reward Environment"]
        AST["Sandboxed AST Math Oracle\n(100% Exact Decimal Evaluation)"]
        CF["Counterfactual Mutation Engine\n(Surgical Date, Entity, & Polarity Flips)"]
        Truth["Deterministic Ground Truth\n(Legal, Financial, & Audit Invariants)"]
    end

    subgraph Agent ["Apex-748M Policy π_θ"]
        Doc["Context (up to 512K tokens)\n+ Proposition / Candidates"] --> Enc["Tier 1 + Tier 2 Encoder"]
        Enc --> Heads["Decision Heads (Noul, Choice, Score)"]
        Heads --> Decision["Categorical Decision & Confidence\n(Zero Token Generation)"]
    end

    Decision --> Env
    Env --> Reward["Objective Verifiable Reward R\n- Symbolic Correctness (+1.0 / -2.5)\n- Counterfactual Invariance (+1.0 / -3.0)\n- Probability Calibration Penalty"]
    Reward --> PolicyGrad["GRPO Advantage Calculation & Gradient Update\n(Group Size G = 4, No Critic Network)"]
    PolicyGrad --> Enc
```

### 7.2 Two-Phase Training Protocol: SFT Warmup & GRPO Policy Optimization
To prevent policy collapse from random initialization on 750M heads, training proceeds in two disciplined phases:

#### Phase 1: Deterministic Supervised Warmup (SFT Seed)
* **Token Budget:** 300M–2.0B tokens (500M tokens for prototype seed, up to 2.0B tokens for full foundation run).
* **Objective:** Standard Cross-Entropy trained directly on synthetically generated, 100% verified ground-truth data (AST-solved balance sheets, programmatic legal contracts, entity relation anchors).
* **Zero Teacher LLM:** Ground-truth labels are generated programmatically by deterministic code compilers and arithmetic oracles, not LLMs.
* **Role:** Establishes stable syntactic representations, initializes positional RoPE baselines, and freezes into the reference policy $\pi_{\text{ref}}$ to anchor the KL penalty during RLVR.

#### Phase 2: Group Relative Policy Optimization (GRPO)
While traditional RLVR (e.g. DeepSeek-R1) optimizes sequential next-token probabilities, Apex-748M applies policy optimization over **categorical decision heads and continuous confidence estimates**:

1. **Policy Distribution ($\pi_\theta$):**
   * For binary belief (Noul): Bernoulli policy $\pi_\theta(y | x, q) = \sigma(z_{\text{noul}})^y (1 - \sigma(z_{\text{noul}}))^{1-y}$.
   * For candidate choice: Categorical policy $\pi_\theta(i | x, \mathcal{C}) = \frac{\exp(s_i)}{\sum_{j=1}^K \exp(s_j)}$.
   * For ordinal scoring: Monotonic cumulative link PMF over discrete tiers $k \in \{0, \dots, K\}$.
2. **Group Sampling ($G = 4$):**
   To avoid the memory overhead of a separate 750M parameter value network (critic), advantages are computed across a group of $G=4$ mutated variations per document context (1 baseline query, 1 counterfactual perturbation, 1 numerical distractor, 1 entity swap):
   $$A_i = \frac{R_i - \text{mean}(\{R_1 \dots R_G\})}{\text{std}(\{R_1 \dots R_G\}) + \epsilon}$$
3. **Objective Function:**
   $$\mathcal{L}_{\text{RLVR}}(\theta) = -\mathbb{E} \left[ \log \pi_\theta(y_i | x) \cdot A_i \right] + \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$$

### 7.3 Why RLVR Outperforms Supervised Cross-Entropy on Decision Heads
Standard Cross-Entropy ($\mathcal{L}_{\text{CE}} = -\log p(y^*)$) pushes probability mass onto isolated targets while treating all incorrect distractors equally. RLVR with group sampling resolves three fundamental limitations of CE:
1. **Multi-Constraint Verifiable Composition:** In enterprise compliance and financial analysis, a verdict requires both semantic entailment and exact arithmetic calculation. Standard CE rewards lucky guesses where the model produced an erroneous intermediate calculation. RLVR verifies the intermediate AST derivation and only awards positive reward when both the math and categorical verdict are provably sound.
2. **Contrastive Margin over Counterfactuals:** Evaluating $G=4$ candidate permutations simultaneously forces the policy gradient to actively push down distractors that share surface keywords but violate causal rules, creating robust decision margins that prevent adversarial susceptibility.
3. **Dynamic Calibration Shaping:** The continuous Brier reward directly penalizes overconfident errors during policy exploration, counteracting the well-known tendency of Cross-Entropy to produce overconfident, miscalibrated probability spikes.

### 7.4 Multi-Component Verifiable Reward Function & Calibration Disambiguation
The reward $R$ is strictly computed by programmatic oracles with zero human or LLM subjectivity:
$$R = R_{\text{symbolic}} + R_{\text{invariance}} + R_{\text{calibration}}$$

| Reward Component | Objective & Verification Condition | Value Assigned |
| :--- | :--- | :--- |
| **$R_{\text{symbolic}}$ (Exact Math)** | Output matches deterministic AST evaluation of numerical expressions. Contradictions or numerical inaccuracies receive an asymmetric penalty. | **$+1.0$** (Correct)<br>**$-2.5$** (Incorrect/Arithmetic Fault) |
| **$R_{\text{invariance}}$ (Causal Logic)** | Evaluated on paired counterfactual documents $(x^+, x^-)$. If surgical changes to dates/counterparties flip the decision as required by logic, reward is granted. If decision remains invariant to causal flips (lazy keyword matching), penalty is applied. | **$+1.0$** (Correct Flip)<br>**$-3.0$** (Invariant Failure) |
| **$R_{\text{calibration}}$ (Honest Confidence)** | Penalizes overconfident errors via strictly proper scoring (Brier penalty): $-(\hat{p} - \mathbb{I}(\text{correct}))^2$. | **$0.0 \to -1.0$** (Continuous) |

#### Optimization vs. Post-Hoc Calibration Disambiguation
* **During RLVR Training:** Temperatures are held fixed ($T_{\text{noul}} = 1.0, \tau_{\text{choice}} = \sqrt{d}$). The Brier reward $R_{\text{calibration}}$ shapes the raw weight projections and logit margins directly inside $\pi_\theta$.
* **Post-Hoc Verification & Deployment Gating:** After RLVR converges, all model weights $\theta$ are frozen. Scalar temperature scaling is fitted on a held-out validation split ($N \ge 10,000$ samples) solely to minimize residual global logit variance. **No post-hoc method guarantees out-of-distribution calibration.** Therefore, Expected Calibration Error (ECE) is reported across 10 uniform confidence bins on out-of-distribution evaluation sets, and automated deployment gating requires validation ECE $\le 0.03$ and selective prediction abstention when model uncertainty exceeds threshold $\theta_{\text{abstain}}$.

### 7.5 Training Economics & Resumable Schedule (Accounting for GRPO Multiplier)

#### FLOP Formulation with GRPO Group Size $G = 4$
Standard supervised training requires $6N$ FLOPs per token ($2N$ forward $+ 4N$ backward). Under GRPO with group size $G=4$:

* **Cached Reference Policy Assumption (Production Baseline — $26N$ FLOPs):**
  Because the document context is identical across the $G=4$ query variations and only queries/prompts vary, reference policy latent representations $\pi_{\text{ref}}(x)$ are evaluated once and cached:
  * Policy rollouts (forward): $G \times 2N = 8N$ FLOPs.
  * Cached reference policy forward pass: $1 \times 2N = 2N$ FLOPs.
  * Policy backward passes: $G \times 4N = 16N$ FLOPs.
  * **Total per Prompt Group:** $(G + 1) \cdot 2N + G \cdot 4N = 10N + 16N = \mathbf{26N\text{ FLOPs}}$ ($\mathbf{\approx 4.33\times}$ supervised compute).
  * *Compute Result:* Tier-1 runs in $\mathbf{\approx 40.9\text{ hours}}$ on TPU v3-8 ($147\text{ TFLOPs}$ sustained).

* **Uncached Reference Policy Fallback ($32N$ FLOPs):**
  If reference states are not cached and re-computed across all $G=4$ variations ($G \times 2N = 8N$ ref FLOPs):
  * Total per Prompt Group: $G \cdot 2N (\text{policy}) + G \cdot 2N (\text{ref}) + G \cdot 4N (\text{backward}) = \mathbf{32N\text{ FLOPs}}$ ($\mathbf{\approx 5.33\times}$ supervised compute).
  * *Compute Result:* Tier-1 requires $2.24\text{ exaFLOPs (SFT)} + 23.90\text{ exaFLOPs (GRPO)} = \mathbf{26.14\text{ exaFLOPs}} \implies \mathbf{\approx 49.4\text{ hours}}$ on TPU v3-8 ($147\text{ TFLOPs}$ sustained).

#### Two-Tier Execution Strategy

| Tier | Token Volume & Phase Breakdown | Compute Target | Use Case |
| :--- | :--- | :--- | :--- |
| **Tier 1: Colab RLVR Prototype** | **1.5 Billion Tokens Total**<br>• Phase 1: 500M SFT warmup tokens ($2.24\text{ exaFLOPs}$)<br>• Phase 2: 1.0B GRPO tokens ($19.42\text{ exaFLOPs}$, 26N cached)<br>• **Total: $21.66\text{ exaFLOPs}$** | **~40.9 hours on TPU v3-8**<br>(147 TFLOPs sustained at 35% MFU; resumable across 6–7 Colab Pro sessions via verified Google Drive checkpointing) | Verifies SFT warmup convergence, GRPO advantage stability with $G=4$, AST math oracle rewards, 512K context memory retention, and zero-token decision accuracy. |
| **Tier 2: Production RLVR Foundation Run** | **10.0 Billion Tokens Total**<br>• Phase 1: 2.0B SFT warmup tokens ($8.96\text{ exaFLOPs}$)<br>• Phase 2: 8.0B GRPO tokens ($155.38\text{ exaFLOPs}$, 26N cached)<br>• **Total: $164.34\text{ exaFLOPs}$** | **~3.2 days on TPU v3-32 pod**<br>(4 hosts / 32 cores, 588 TFLOPs sustained at 35% MFU; or ~12.9 days on a dedicated single Cloud TPU v3-8 instance at 147 TFLOPs sustained) | Full from-scratch convergence reaching superhuman decision accuracy, proven mathematical robustness, and enterprise ECE deployment gating. |

#### Empirical Evaluation Harness (JEV/CLEF Protocol)
To evaluate the model objectively without unsubstantiated claims:
1. **JEV Parity Suite:** 1,000 matched enterprise decision items measuring macro-F1 and ECE.
2. **Needle-in-a-Haystack:** 512K context test with needles placed at 5%, 25%, 50%, 75%, and 95% depth.
3. **Counterfactual Invariant Test:** Verifies that swapping entity names or flipping dates flips binary decisions.
4. **Symbolic Exactness Test:** 10,000 synthetically generated financial expressions evaluated for exact match.

---

## 8. Secure Google Colab & TPU Execution Workflow

### 8.1 Secure Authentication Workflow
```python
# 1. For public repository (no authentication required):
!git clone https://github.com/Raj1232309/apex-decision.git
%cd apex-decision

# 2. For private repository (uses Colab Secrets without command-line leaking):
from google.colab import userdata
import os

token = userdata.get('GITHUB_TOKEN')
cred_path = os.path.expanduser('~/.git-credentials')
with open(cred_path, 'w') as f:
    f.write(f"https://x-access-token:{token}@github.com\n")
os.chmod(cred_path, 0o600)  # Restrict permissions
os.system("git config --global credential.helper store")
os.system("git clone https://github.com/Raj1232309/apex-decision.git")
%cd apex-decision
```

### 8.2 Verified Checkpointing Protocol
```python
import os
import shutil
import hashlib
import torch
from google.colab import drive

def save_verified_checkpoint(model, optimizer, step, save_dir="/content/drive/MyDrive/apex_checkpoints"):
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive', force_remount=False)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Save locally to /tmp first (synchronous local write avoids FUSE timeouts)
    local_path = f"/tmp/apex_step_{step}.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict()
    }, local_path)
    
    # 2. Compute local SHA-256
    sha256 = hashlib.sha256()
    with open(local_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    local_hash = sha256.hexdigest()
    
    # 3. Check disk space on Google Drive
    total, used, free = shutil.disk_usage(save_dir)
    if free < os.path.getsize(local_path) * 2:
        raise OSError("INSUFFICIENT_GOOGLE_DRIVE_SPACE")
        
    # 4. Synchronously copy and verify remote hash
    target_path = os.path.join(save_dir, f"apex_step_{step}.pt")
    shutil.copyfile(local_path, target_path)
    
    remote_sha256 = hashlib.sha256()
    with open(target_path, "rb") as f:
        while chunk := f.read(65536):
            remote_sha256.update(chunk)
    if remote_sha256.hexdigest() != local_hash:
        raise IOError("CHECKPOINT_CORRUPTION_DETECTED")
        
    print(f"Checkpoint {step} successfully verified and saved: {local_hash[:16]}")
```
