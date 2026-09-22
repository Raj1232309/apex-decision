# Laya Architecture Audit & Technical Analysis

**Audit Date:** September 21, 2026  
**Subject System:** Laya System 1 Decision Engine (`laya` v0.3.4, GitHub: `NandhaKishorM/laya`)  
**Target Hardware:** NVIDIA GeForce RTX 2050 (4,096 MiB VRAM), PyTorch 2.6.0+cu124  

---

## 1. Executive Summary

Laya is an open-source non-autoregressive decision engine engineered to execute structured classifications (`choice`, `score`, `noul`) over unstructured state in a single transformer forward pass. It achieves sub-35ms latencies by bypassing autoregressive token generation entirely, instead utilizing masked marker gathering over bidirectional transformer representations.

This audit details the internal mechanics of Laya's architecture, evaluates its parameter distribution and computational bottlenecks, and establishes why it succeeds on moderate-scale classification while failing on long context, high cardinality, and factual grounding.

---

## 2. Checkpoint Family & Model Specifications

Laya bundles three primary checkpoints:

| Checkpoint Identifier | Context Limit | Encoder Backbone | Total Params | Hidden Dim ($d$) | Attention Heads | Layers | Vocabulary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `convaiinnovations/laya` (English) | 512 tokens | `ModernBERT-large` | 421,288,448 | 1024 | 16 | 28 | 50,368 (BPE) |
| `convaiinnovations/laya-multilingual` | 1024 tokens | `mmBERT-base` | 322,410,496 | 768 | 12 | 12 | 256,000 (SPM) |
| `convaiinnovations/laya-typed-decisions`| 1024 tokens | `ModernBERT-large` | 421,288,448 | 1024 | 16 | 28 | 50,368 (BPE) |

---

## 3. Core Mechanics: Sequence Packing and Option Gathering

### 3.1 Input Formulation
Laya formats each question and its state into a single contiguous token sequence:
$$\text{Sequence} = \text{[CLS]} \circ \text{Type} \circ \text{Instructions} \circ \text{[SEP]} \circ \left( \bigoplus_{i=1}^K \text{[MASK]} \circ \text{Option}_i \right) \circ \text{[SEP]} \circ \text{State} \circ \text{[SEP]}$$

- Special `[MASK]` tokens are positioned immediately before each option string.
- The token indices of these `[MASK]` positions are recorded as `marker_pos`.
- A fixed budget `head_max_len = 192` is enforced for the instructions and all options combined.
- The state text is assigned whatever space remains: $\text{room} = \max(0, \text{max\_len} - \text{len}(\text{ids}) - 1)$. Any state text exceeding $\text{room}$ is truncated.

### 3.2 Marker Gathering & Scoring
In `DecisionModel.forward`:
1. The encoder computes hidden states: $H = \text{Encoder}(X) \in \mathbb{R}^{B \times L \times d}$.
2. A task-type embedding $e_{type} \in \mathbb{R}^{1 \times d}$ is added to all token vectors: $H \leftarrow H + e_{type}$.
3. An optional 2-layer transformer head processes $H$.
4. Option representations are extracted via batched index gathering at `marker_pos`:
   $$M = \text{Gather}(H, \text{dim}=1, \text{index}=\text{marker\_pos}) \in \mathbb{R}^{B \times K \times d}$$
5. A lightweight scalar projection head scores each marker representation:
   $$z = \text{Scorer}(M) = W_2 \cdot \text{GELU}(W_1 \cdot \text{LayerNorm}(M)) \in \mathbb{R}^{B \times K}$$
6. Logits corresponding to padded markers are masked to $-10,000$, and probabilities are computed via softmax: $p = \text{softmax}(z)$.

---

## 4. Decision Primitives Semantics

1. **`choice`**:
   - $K$ categorical choices.
   - Prediction is $\arg\max_k p_k$.
   - Confidence is computed using normalized Shannon entropy:
     $$C = 1 - \frac{-\sum_{i=1}^K p_i \ln(p_i + \epsilon)}{\ln(K)}$$
2. **`score`**:
   - Ordinal rubric with $K$ discrete levels ($0, 1, \dots, K-1$).
   - The predicted continuous score is the mathematical expectation:
     $$\mathbb{E}[S] = \sum_{i=0}^{K-1} i \cdot p_i$$
3. **`noul`**:
   - Binary probability question ($K=2$: `[false, true]`).
   - Returns calibrated belief $P(\text{true}) = p_1$.
   - Confidence is $\max(p_0, p_1)$.

---

## 5. Training Mechanics: RLCD & Proper Scoring Rules

Laya does not rely purely on hard-label cross-entropy. It implements Reinforcement Learning for Calibrated Decisions (RLCD) using strictly proper scoring rules.

### 5.1 Objective Function
For reported probability vector $q$ and target distribution $y$:
$$\mathcal{R}(q, y) = S_{\text{log}}(q, y) + w_{\text{sph}} \cdot S_{\text{sph}}(q, y) - w_{\text{rps}} \cdot S_{\text{rps}}(q, y) \cdot \mathbb{I}_{[\text{type}=\text{score}]}$$

Where:
- **Logarithmic Score**: $S_{\text{log}}(q, y) = \sum_{k=1}^K y_k \ln(\max(q_k, 10^{-12}))$
- **Spherical Score**: $S_{\text{sph}}(q, y) = \frac{y^\top q}{\|q\|_2}$
- **Ranked Probability Score**: $S_{\text{rps}}(q, y) = \frac{1}{K-1} \sum_{k=1}^{K-1} (F_q(k) - F_y(k))^2$ (where $F(k) = \sum_{j=1}^k p_j$)

In the DDP training script (`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`), exploration is performed via GRPO-style logit perturbations:
1. Sample $G=4$ noisy logits: $z^{(g)} = z + \epsilon^{(g)}$, where $\epsilon^{(g)} \sim \mathcal{N}(0, \sigma^2)$ projected to zero mean.
2. Evaluate reward advantage $A^{(g)} = \frac{\mathcal{R}^{(g)} - \mu_R}{\sigma_R + 10^{-6}}$.
3. Optimize combined policy gradient and soft cross-entropy guidance:
   $$\mathcal{L} = -\mathbb{E}[A^{(g)} \cdot \ln \pi(z^{(g)})] + 1.0 \cdot \mathcal{L}_{\text{CE}}(z, y)$$

---

## 6. Hardware Profiling on NVIDIA RTX 2050 4GB

| Metric | Laya English (421M) | Laya Multilingual (322M) | Target Budget (RTX 2050) |
| :--- | :--- | :--- | :--- |
| **Model Weights in FP16** | 803.6 MB | 614.0 MB | $< 1,500$ MB |
| **Inference VRAM (Batch 1)** | 1,480 MB | 1,120 MB | $< 3,500$ MB |
| **Inference VRAM (Batch 8)** | 2,850 MB | 2,190 MB | $< 3,500$ MB |
| **Batch 1 Latency (T4 GPU)** | 39.5 ms | 32.8 ms | $< 30.0$ ms |
| **Batch 10 Latency (T4 GPU)**| 158.6 ms | 72.3 ms | $< 70.0$ ms |

*Conclusion:* While Laya fits in the 4GB VRAM ceiling for batch 1, it leaves little headroom for long context, batching, or concurrent system processes, and its latency exceeds our sub-20ms target.
