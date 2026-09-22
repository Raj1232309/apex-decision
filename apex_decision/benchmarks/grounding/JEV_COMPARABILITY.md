# TypeSafe Jev Comparability & Benchmark Integrity Analysis

**Document Status:** Official Methodological Position  
**Author:** ApexDecision Research & Evaluation  
**Reference Target:** TypeSafe Jev (Published 96.3% Accuracy on 108 Grounding Claims)

---

## 1. Context & Methodological Policy

In previous reports, ApexDecision compared internal prototype results against a 67.8% figure for TypeSafe Jev. That comparison was fundamentally invalid because the 67.8% metric was derived from a multi-stage workflow execution benchmark rather than Jev's reported grounding verification engine.

Furthermore, TypeSafe Jev has publicly reported:
> **96.3% accuracy on 108 grounding claims across six domains.**

To ensure rigorous scientific integrity, this document establishes what is publicly known about Jev's benchmark, what remains proprietary or unknown, and the exact constraints governing any comparisons.

---

## 2. Publicly Documented Information About Jev

Based on published technical disclosures and documentation from TypeSafe:
* **Benchmark Size:** Exactly 108 claims.
* **Domain Scope:** 6 primary enterprise domains (Science, Medicine, Finance, Technology, Law/Policy, Business).
* **High-Level Task:** Factual claim verification / grounding determination given source document context.
* **Reported Overall Grounding Accuracy:** 96.3% (approximately 104 out of 108 claims correctly verified).
* **Reported Latency Profile:** p50 latency of ~120 ms, p99 latency of ~480 ms (cloud-hosted service via HTTP/REST).

---

## 3. What Is Unknown / Proprietary

The following critical experimental details are **not publicly available** in Jev's published disclosures:
1. **Exact Test Samples:** The specific 108 prompt-claim pairs, source documents, and annotations are proprietary and withheld from public release.
2. **Supported vs. Unsupported Class Balance:** The exact proportion of true claims versus false claims in Jev's 108-claim set is not disclosed.
3. **Distribution of Adversarial Near-Misses:** The exact percentage of subtle adversarial perturbations (e.g. numerical tampering vs entity substitution vs direct contradictions) is unspecified.
4. **Difficulty Stratification:** Whether claims were stratified by syntactic difficulty, sentence length, or multi-hop reasoning complexity is not formally documented.
5. **Confidence Calibration / ECE:** Expected Calibration Error (ECE) or Brier scores for Jev's probability estimates were not published alongside the 96.3% figure.
6. **Hardware & Architecture:** Jev's internal architecture, parameter count, quantization level, and cloud accelerator tier are proprietary.

---

## 4. Benchmark Integrity & Framing Rules

Because Jev's proprietary test data is not publicly available for direct offline execution:
1. **Never Claim Exact Reproduction:**
   Under no circumstances may ApexDecision claim:
   > *"We reproduced Jev's benchmark."*
   
   The official characterization of our 108-claim evaluation suite is strictly:
   > **"Independent grounding benchmark inspired by the published Jev evaluation."**

2. **No Fabricated Predictions:**
   We do not manufacture or simulate Jev's individual prediction outputs. Where direct API comparison is unavailable, Jev's published 96.3% figure is cited as an external published reference benchmark.

3. **Separate Benchmark Design:**
   We maintain two separate evaluation tracks:
   * **Development & Scale Track:** A comprehensive 5,308-sample benchmark (with 600 held-out test examples across 8 domains and 20 granular failure modes) used for training, hard-negative mining, calibration, and ablation studies.
   * **Jev-Style Reporting Track:** A strictly isolated 108-claim suite matching Jev's high-level structure (6 domains, 18 claims per domain, 4 difficulty tiers: Tier 1 Direct, Tier 2 Paraphrase, Tier 3 Adversarial Near-Match, Tier 4 Complex Logic). This split is evaluated once at the conclusion of the study and is never seen during training or hyperparameter tuning.
