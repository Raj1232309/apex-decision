# Laya Integration Map: Teacher Supervision Matrix for ApexStudent

**Date:** September 21, 2026  
**Objective:** Map teacher representations, probability distributions, and scoring rules from Laya into the from-scratch `ApexStudent` training pipeline.

---

## 1. System Roles: Teacher vs Student Separation

```
+-------------------------------------------------------------+
|                      LAYA (TEACHER)                         |
|  - ModernBERT-large (421M) / mmBERT-base (322M)             |
|  - Role: Offline Teacher, Research Baseline, Soft Targets   |
|  - Runtime Dependency: ZERO (Not called at production time) |
+-------------------------------------------------------------+
                               |
                               | Distillation Supervision:
                               | - Soft Probabilities (T=2.0)
                               | - Marker Logit Margins
                               | - Calibration Confidence Targets
                               | - RLCD Proper Reward Signals
                               v
+-------------------------------------------------------------+
|                   APEXSTUDENT (OUR MODEL)                   |
|  - Independent 38.5M Transformer Backbone (From Scratch)    |
|  - Custom 16,384-token BPE Tokenizer                        |
|  - 256K - 1M Hierarchical Streaming Context Architecture    |
|  - 2-Stage Hierarchical Choice Head (Up to 255 options)     |
|  - Deterministic Provenance & Polarity Verification Layer   |
|  - Zero Pretrained Weights, Zero External Runtime Calls     |
+-------------------------------------------------------------+
```

---

## 2. Distillation Transfer Matrix

| Target Task / Capability | Teacher Signal (Laya) | Distillation Method | Student Target Head |
| :--- | :--- | :--- | :--- |
| **Categorical Choice** | Logit distribution across options $z_k$ and soft probabilities $p_k$ | KL-Divergence with Temperature Scaling ($T=2.0$) | `ApexStudent.choice_head` (Hierarchical 2-stage) |
| **Continuous & Ordinal Score** | Expected score $\sum i \cdot p_i$ and probability distribution over levels | Ranked Probability Score (RPS) Loss + Soft CE | `ApexStudent.score_head` |
| **Boolean & Noul Grounding** | Calibrated $P(\text{true})$ and confidence $C(p)$ | Spherical Scoring + Brier-Regularized Binary Cross-Entropy | `ApexStudent.noul_head` (3-way: SUPPORTED, CONTRADICTS, INSUFFICIENT) |
| **Uncertainty & Abstention** | Normalized Shannon entropy $1 - H(p)/\ln(k)$ | Policy Gradient RLCD Advantage Matching | `ApexStudent.act_head` (Abstention Gate) |
| **Multilingual Script Routing** | Script code-point ranges and stopword heuristics (`laya/lang.py`) | Pure Python Pre-flight Regex & Unicode Filter | `ApexStudent.router` |

---

## 3. Loss Formulation for Student Training

The student training loss combines 4 complementary supervision objectives:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{hard}} + \beta \mathcal{L}_{\text{distill}} + \gamma \mathcal{L}_{\text{proper}} + \delta \mathcal{L}_{\text{abstain}}$$

1. **Hard Cross-Entropy ($\mathcal{L}_{\text{hard}}$)**:
   Anchors the student directly to ground-truth labels on verified domain tasks:
   $$\mathcal{L}_{\text{hard}} = -\sum_{k=1}^K y_k \ln(q_k)$$

2. **Soft Label Distillation ($\mathcal{L}_{\text{distill}}$)**:
   Transfers teacher uncertainty and inter-class similarity from Laya:
   $$\mathcal{L}_{\text{distill}} = T^2 \cdot \text{KL}\left(\text{softmax}\left(\frac{z_{\text{teacher}}}{T}\right) \;\Big\|\; \text{softmax}\left(\frac{z_{\text{student}}}{T}\right)\right)$$
   Where $T = 2.0$.

3. **RLCD Proper Scoring Rule Loss ($\mathcal{L}_{\text{proper}}$)**:
   Incentivizes calibrated, honest probability reports:
   $$\mathcal{L}_{\text{proper}} = -\mathbb{E}\left[ S_{\text{log}}(q, y) + 0.75 \cdot S_{\text{sph}}(q, y) - 1.0 \cdot S_{\text{rps}}(q, y) \cdot \mathbb{I}_{[\text{score}]} \right]$$

4. **Abstention & Grounding Loss ($\mathcal{L}_{\text{abstain}}$)**:
   Penalizes confident predictions when evidence is missing or adversarial:
   $$\mathcal{L}_{\text{abstain}} = \text{Brier}(q, y) + \lambda_{\text{ent}} \cdot \max(0, H_{\text{target}} - H(q))$$

---

## 4. Architectural Enhancements Independent of Teacher

Where Laya fails, the student does not copy teacher weaknesses; instead, independent architectural mechanisms take over:

1. **Context Expansion**: When state exceeds 1,024 tokens, Laya truncates. The student passes the full text through the `HierarchicalContextEngine` (512-token chunking, salience scoring, recurrent memory token aggregation) to feed top evidence windows into the student encoder.
2. **High-Cardinality Choice**: When options exceed 20, Laya collapses. The student passes candidate embeddings through a coarse clustering gate, selecting top-16 candidates for fine-grained marker gathering.
3. **Deterministic Safety Guard**: The student's predictions are passed through `ProvenanceVerifier` and `NumericPolarityGuard` to guarantee zero invented citations and zero inverted polarities.
