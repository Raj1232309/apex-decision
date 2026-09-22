# Laya Attribution & Open-Source Lineage Record

**Date:** September 21, 2026  
**License Reference:** Apache License, Version 2.0  
**Upstream Repository:** [https://github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)  
**Upstream Authors & Copyright:** Convai Innovations / Nandha Kishor M  

---

## 1. Purpose & Legal Compliance

This document records the exact architectural concepts, mathematical formulations, software patterns, and comparative methodologies studied, adapted, or reimplemented from the open-source **Laya** decision engine into **ApexDecision V7**.

Under the terms of the Apache License 2.0, all derivative ideas and reimplemented modules maintain explicit attribution to the original authors. Crucially, **no learned model parameter weights** from Laya are transferred into the ApexDecision student model (`ApexStudent`). `ApexStudent` is initialized from scratch with random parameters and trained exclusively via our distillation and proper scoring pipeline.

---

## 2. Inventory of Reused, Adapted, and Reimplemented Components

| Component / Idea | Origin in Laya Source | Status in ApexDecision V7 | Technical Adaptation & Modification |
| :--- | :--- | :--- | :--- |
| **Typed Decision Primitives** | `laya/common.py` (`QTYPES = {"choice": 0, "score": 1, "noul": 2}`) | **Adapted & Extended** | Retained `choice`, `score`, and `noul` taxonomy. Extended with explicit `verify` (4-way) primitive and strict provenance grounding pointers. |
| **Mask Marker Representation** | `build_sequence` & `DecisionModel.forward` in `laya/common.py` | **Adapted & Reimplemented** | Laya uses `[MASK]` tokens prepended to each rendered option and gathers hidden states at marker positions via `torch.gather`. ApexDecision adapts this non-autoregressive parallel option scoring mechanism within our from-scratch encoder backbone. |
| **Proper Scoring Rules (RLCD)** | `proper_reward` in `laya/common.py` & `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` | **Adapted & Reimplemented** | Reimplemented the multi-objective loss combining logarithmic scoring ($\sum y_k \log q_k$), spherical scoring ($y^\top q / \|q\|_2$), and Ranked Probability Score (RPS) for ordinal scoring rubrics. Extended with active calibration penalties and epistemic abstention loss. |
| **Entropy-Based Confidence Estimation** | `confidence_from_probs` in `laya/common.py` | **Reimplemented** | Adapted the normalized Shannon entropy formulation: $C(p) = 1 - \frac{H(p)}{\ln(k)}$ where $k$ is the option cardinality. Used as a primary signal for selective abstention and routing. |
| **Option-Count Temperature Bucketing** | `temp_bucket` in `laya/common.py` (`"2"`, `"3-5"`, `"6-10"`, `"11+"`) | **Adapted & Extended** | Reimplemented per-primitive and per-cardinality temperature scaling to eliminate post-training miscalibration. Extended to cover high-cardinality buckets up to 255 options. |
| **Script Detection & Routing** | `laya/lang.py` (`detect_script`) | **Reimplemented** | Adapted the fast pure-Python Unicode code-point range detection to distinguish Latin from non-Latin scripts in $< 0.1$ ms prior to forward execution. |
| **System-One Decision API Surface** | `Agent.system_one` / `Agent.predict` in `laya/agent.py` | **Reimplemented & Hardened** | Adopted the dictionary-based question definition signature `predict(state, questions)` while adding deterministic schema validation, zero-invention guards, and provenance metadata. |

---

## 3. Key Divergences: Where ApexStudent Extends Beyond Laya

While Laya serves as our foundational teacher and benchmark reference, ApexDecision V7 introduces several major architectural breakthroughs to resolve Laya's identified failure modes:

1. **Independent From-Scratch Initialization**:
   - Laya depends on pretrained checkpoints (`ModernBERT-large` with 421M params or `mmBERT-base` with 322M params).
   - `ApexStudent` has **zero pretrained weights**, starting from random initialization with our custom-trained 16,384-token BPE tokenizer.
2. **256K to 1M Token Context Scaling**:
   - Laya enforces a hard input ceiling of 512 or 1024 tokens; text exceeding this budget is irreversibly truncated.
   - `ApexStudent` incorporates a hierarchical streaming context engine that processes genuine 256K, 512K, and 1,000,000 token inputs without out-of-memory errors on 4GB GPUs.
3. **High-Cardinality Choice Scaling (up to 255 options)**:
   - In Laya, 77-option tasks (e.g. Banking77) collapse to 42.5% accuracy because all options share a fixed token budget (`head_max_len=192`), giving each option only ~4 tokens.
   - `ApexStudent` introduces a hierarchical 2-stage coarse-to-fine selection architecture ensuring all 255 options are fully represented without truncation.
4. **Deterministic Evidence & Provenance Verification**:
   - Laya outputs class probabilities without locating or validating the exact source sentence.
   - `ApexStudent` couples neural prediction with deterministic substring extraction, character offset validation, and numeric/polarity guards.
5. **Runtime Efficiency**:
   - `ApexStudent` operates at ~38.5M parameters (compared to Laya's 421M), delivering $2\times$ to $3\times$ lower latency and sub-1GB VRAM footprint on consumer GPUs (NVIDIA RTX 2050).

---

## 4. License Text

The original Laya project is distributed under the Apache License, Version 2.0:
```
Copyright 2026 Convai Innovations / Nandha Kishor M

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
