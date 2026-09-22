# Attribution Notice: Laya & ConvAI Innovations

**Project:** ApexDecision  
**Reference Repository:** `laya` (`convaiinnovations/laya`)  
**License:** Apache License 2.0 (January 2004)  
**Copyright:** Copyright (c) 2025 ConvAI Innovations Inc.

---

## 1. Overview & Research Attribution

ApexDecision gratefully acknowledges the pioneering research and open-source contributions of ConvAI Innovations on the **Laya System 1 Decision Model** family:
- `convaiinnovations/laya` (ModernBERT-large 421M, 512 tokens)
- `convaiinnovations/laya-multilingual` (mmBERT-base 322M, 1024 tokens)
- `convaiinnovations/laya-typed-decisions` (ModernBERT-large 421M, 1024 tokens)

In accordance with Section 4 of the Apache License 2.0, this document provides explicit attribution for architectural patterns, algorithmic formulations, and typed decision mechanisms adapted or referenced in ApexDecision.

---

## 2. Adapted Concepts and Algorithmic Formulations

The following core algorithmic concepts from Laya have informed the design of ApexDecision:

1. **Typed Decision Primitives:**
   - The separation of decision outputs into categorical choice (`choice`), continuous regression / expected value rubric scoring (`score`), and nullary / binary factual assertions (`noul`).
   - Adapted in `apex_decision/models/apex_student/model.py` and `apex_decision/inference/apex_student/runtime.py`.

2. **RLCD & Proper Scoring Rules:**
   - Utilization of strictly proper scoring rules (Log-score, Spherical scoring, and Ranked Probability Score / RPS) alongside KL divergence distillation from soft teacher distributions.
   - Adapted in `apex_decision/training/rlcd/losses.py`.

3. **Multi-Bucket Temperature Calibration:**
   - Option-count and primitive-type aware temperature scaling to adjust predicted confidence distributions.
   - Adapted in `apex_decision/training/calibration/calibrate.py`.

4. **Workflow Presets & Problem Signatures:**
   - Question schemas for ticket triage, threat filtering, LLM guardrails, and model routing.
   - Adapted in `apex_decision/evaluation/` and `apex_decision/training/data/capability_pools.py`.

---

## 3. Strict Pretrained Weight Independence Invariant

**CRITICAL COMPLIANCE NOTICE:**
- **Zero Pretrained Weights Copied:** Under no circumstances are pretrained Laya or ModernBERT model weights copied into or distributed with the ApexStudent inference artifacts.
- **Independent Initialization:** All ApexStudent models (`ApexStudent-Baseline`, `ApexStudent-50M`, `ApexStudent-100M`) are initialized strictly from scratch with random Xavier initialization and trained independently.
- **Offline Teacher Reference Only:** Laya is employed purely as an offline research reference, evaluation benchmark, and soft target distillation distribution provider during training.
- **Zero Inference Dependency:** The ApexStudent production runtime has zero runtime dependencies on Laya, HuggingFace Transformers, or any external model server.
