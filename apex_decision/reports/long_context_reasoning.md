# ApexDecision V7.2: Long-Context Multi-Hop Reasoning Report

**Evaluation Date:** September 21, 2026  
**Hardware:** NVIDIA GeForce RTX 2050 (4,096 MiB VRAM), PyTorch 2.6.0+cu124  
**Benchmark Suite:** `apex_decision/evaluation/long_context_reasoning_suite.py`  
**Test Scales:** 256,000 Tokens | 512,000 Tokens | 1,000,000 Tokens  

---

## 1. Executive Summary: Decoupling Retrieval from Reasoning

In previous evaluations (V7.1), long-context benchmarks demonstrated needle retrieval (locating isolated strings inside massive documents). However, real-world decision workflows require far more than finding needles: they require **multi-hop reasoning**, **temporal supersession**, and **contradiction resolution** across widely distributed premises.

This audit evaluates long-context performance across **five distinct, decoupled stages**:
1. **Stage A (Retrieval):** Locating distributed premises across hundreds to thousands of distractor blocks.
2. **Stage B (Comprehension):** Accurately extracting dates, entities, and numeric parameters from isolated retrieved blocks.
3. **Stage C (Multi-hop Reasoning):** Synthesizing distributed premises (Premise 1 in Block $i$ + Premise 2 in Block $j$) to deduce unstated composite facts.
4. **Stage D (Contradiction Resolution):** Recognizing when a later amendment or addendum supersedes and invalidates an earlier clause.
5. **Stage E (Final Decision):** Emitting the final grounded decision (`SUPPORTED` / `NOT_SUPPORTED`) with exact character-level provenance spans.

---

## 2. Experimental Setup & Distributed Document Design

To eliminate template repetition and artificial clues, we constructed a realistic multi-million-character enterprise corporate corpus containing procurement policies, ERP incident reports, logistics telemetry, and audit logs.

Within this corpus, four interconnected facts were distributed at distant relative depths ($10\%$, $40\%$, $70\%$, and $95\%$ of total document length):
- **Fact 1 (10% depth):** *"Section 4.2 of the Master Services Agreement establishes that Project Titan was assigned an initial capital expenditure budget of $45.0M in January 2022."*
- **Fact 2 (40% depth):** *"Amendment 3 executed on July 14, 2023 officially increased Project Titan's authorized budget by $18.5M for European cloud infrastructure."*
- **Fact 3 (70% depth):** *"The FY2024 Year-End Financial Audit recorded actual cumulative expenditure for Project Titan at $61.2M as of November 2024."*
- **Fact 4 / Superseding Addendum (95% depth):** *"Superseding Addendum: On December 15, 2024, the Executive Committee approved an emergency write-down of $5.0M for Project Titan, establishing the final certified expenditure at $56.2M."*

---

## 3. Empirical Results Across Context Scales

All tests were executed on the target **NVIDIA GeForce RTX 2050 4GB GPU**.

### Stage-by-Stage Scoreboard:

| Context Scale | Actual Tokens | Document Chars | Stage A: Retrieval Recall | Stage B: Comprehension | Stage C: Multi-Hop Reasoning | Stage D: Contradiction Resolution | Stage E: Final Decision | Retrieval Latency | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256K (Student)** | 192,854 | 1,112,754 | **100.0% (4/4)** | **PASSED** | **FAILED** | **PASSED** | **PASSED (Verified)** | 106.1 ms | 2,567.9 MB |
| **256K (Laya)** | 192,854 | 1,112,754 | 0.0% (0/4) | Untested (Truncated) | FAILED (Truncated) | FAILED (Truncated) | FAILED (Truncated) | — | — |
| **512K (Student)** | 385,708 | 2,225,457 | **100.0% (4/4)** | FAILED | **FAILED** | **PASSED** | FAILED | 190.1 ms | 2,567.9 MB |
| **512K (Laya)** | 385,708 | 2,225,457 | 0.0% (0/4) | Untested (Truncated) | FAILED (Truncated) | FAILED (Truncated) | FAILED (Truncated) | — | — |
| **1M (Student)** | 753,334 | 4,346,551 | **100.0% (4/4)** | FAILED | **FAILED** | FAILED | FAILED | 403.2 ms | 2,567.9 MB |
| **1M (Laya)** | 753,334 | 4,346,551 | 0.0% (0/4) | Untested (Truncated) | FAILED (Truncated) | FAILED (Truncated) | FAILED (Truncated) | — | — |

---

## 4. Deep-Dive Stage Analysis

### Stage A: Retrieval (The Student's Overwhelming Advantage)
- **ApexStudent:** The `HierarchicalLongContextEngine` chunks the document into 512-token blocks and computes lexical-semantic routing scores in streaming mode. Across all three scales (192K, 385K, and 753K tokens), **all 4 distributed facts were retrieved (100.0% recall)**.
- **Latency & Compression:** Retrieval executed in **106 ms** (256K) to **403 ms** (1M), compressing millions of characters by **92.1x to 361.9x** into a tight evidence window.
- **Laya:** Laya enforces a hard sequence ceiling of `max_len = 512` tokens (~2,000 characters). Because facts are distributed beyond token index 512, Laya truncates them completely ($0.0\%$ recall). Laya cannot process long context without an external chunker.

### Stage B: Comprehension (Isolated Extraction)
- At 256K tokens, the student model correctly extracted Fact 1 (\$45.0M capex) and verified the claim as `SUPPORTED`.
- At 512K and 1M tokens, when 6 top-K blocks were concatenated into the fusion window, the student model's attention was diluted by adjacent distractor paragraphs, producing false negative `NOT_SUPPORTED` verdicts.

### Stage C: Multi-Hop Reasoning (The Architectural Bottleneck)
- **The Task:** Premise 1 states initial budget was \$45.0M. Premise 2 states Amendment 3 increased it by \$18.5M. The test asks: *"Did Amendment 3 increase Project Titan's authorized budget to $63.5M?"*
- **Empirical Finding:** Both ApexStudent and Laya scored **FALSE** across all context scales.
- **Why Multi-Hop Fails in a Single Forward Pass:**
  A 19.5M bidirectional encoder cannot perform composite symbolic arithmetic ($\$45.0\text{M} + \$18.5\text{M} = \$63.5\text{M}$) across separate sentences in a single non-autoregressive step without explicit step-by-step chaining. The model looks for the explicit token "\$63.5M" in the evidence; finding only "\$45.0M" and "\$18.5M", the polarity guard and attention heads flag a numeric discrepancy.

### Stage D: Contradiction Resolution
- At 256K and 512K, the student model successfully resolved the superseding clause: Fact 4 explicitly amended Fact 3, revising certified expenditure from \$61.2M down to \$56.2M. The student correctly supported the post-amendment figure.
- At 1M tokens, distractor noise caused the student to abstain.

### Stage E: Final Decision & Provenance
- At 256K tokens, the final composite decision (*"Did Project Titan's certified expenditure remain within authorized budget?"*) passed with verified character offset provenance citations.
- At 512K and 1M tokens, attention dilution in the fused window prevented the final entailment head from activating.

---

## 5. Architectural Remediation Plan

To elevate 512K and 1M reasoning from partial retrieval to full end-to-end multi-hop correctness:
1. **Isolated Per-Block Extraction (Step 1):** Instead of concatenating 6 blocks into one prompt, execute Stage B independently on each retrieved block (embarrassingly parallel on GPU).
2. **Symbolic Fact Register (Step 2):** Extract structured propositions (`(Project Titan, initial_budget, $45.0M)`, `(Project Titan, amendment_increase, $18.5M)`) into a lightweight epistemic register.
3. **Chained Symbolic Resolver (Step 3):** Run arithmetic and temporal supersession over the register before the final decision pass.
