# ApexDecision v2: High-Precision Entailment & Grounding Architecture Specification

**System**: ApexDecision v2  
**Target Standard**: Defensible, robust claim verification approaching 99.9% accuracy on large frozen evaluation suites ($\ge 10,000$ claims)  
**Primary Optimization Target**: **Direct Entailment Recall** on unseen source documents while preserving zero false positives ($\text{FPR} \le 0.05\%$)  

---

## 1. System Pipeline Architecture

ApexDecision v2 completely departs from treating grounding as an unguided monolithic binary text classifier (`[source] [SEP] [claim]`). Instead, it deploys an end-to-end multi-stage pipeline:

```
                            ┌────────────────────────┐
                            │    SOURCE DOCUMENT     │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │  Sentence Segmenter v2 │
                            │  & Contiguous Chunking │
                            └───────────┬────────────┘
                                        │
                                        ▼
┌───────────────────────┐   ┌────────────────────────┐
│     TARGET CLAIM      │──►│ Hybrid BM25/Dense      │
└───────────┬───────────┘   │ Evidence Retrieval v2  │
            │               └───────────┬────────────┘
            ▼                           │
┌───────────────────────┐               │
│   Claim Decomposer    │               ▼
│  (Atomic Propositions)│──►┌────────────────────────┐
└───────────┬───────────┘   │  Top-k Evidence Spans  │
            │               └───────────┬────────────┘
            │                           │
            ├───────────────────────────┤
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌────────────────────────┐
│  Structured Semantic  │   │   3-Way Cross-Encoder  │
│  Attribute Alignment  │   │   (Entailment Verifier)│
│  (Numbers, Entities,  │   │  [Entail, Contra, Insuf]│
│   Dates, Causality)   │   │                        │
└───────────┬───────────┘   └───────────┬────────────┘
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │   Conservative Decision   │
            │        Aggregator         │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │    Temperature Scaling    │
            │        Calibrator         │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │   SUPPORTED / NOT_SUPP    │
            │   + Cryptographic Proof   │
            └───────────────────────────┘
```

---

## 2. Core Modules & Implementation Details

### A. Sentence Segmenter v2 & Hybrid Evidence Retriever v2 (`evidence_retriever_v2.py`)
- **Limitation of v1**: v1 sources in `test.json` were 1–2 sentences, triggering a pass-through condition (`len(s) <= 2`) that bypassed retrieval entirely.
- **v2 Solution**:
  1. Technical, scientific, medical, and legal sentence boundary disambiguation.
  2. Generates individual sentence spans and contiguous 2-sentence context windows.
  3. Ranks spans via BM25 term frequency overlap augmented by exact entity, numeric, and date salience scores ($1.5\times$ number boost, $1.2\times$ entity boost).
  4. Formulates cohesive evidence contexts, directly enabling **multi-sentence reasoning** where multiple source sentences must be synthesized.

### B. Claim Decomposition Module (`claim_decomposer.py`)
- For complex, compound, or multi-predicate claims, the decomposer splits the claim into atomic propositions $\mathcal{P} = \{p_1, p_2, \dots, p_m\}$ across conjunctions (`and`, `while`, `whereas`, semicolons, relative clauses).
- Extracts entities, numbers, percentages, temporal markers, and negation states per proposition.

### C. Structured Semantic Attribute Alignment (`structured_checks.py`)
- **Permanent Removal of Regex Word Guards**: v1 regex searching for words like `"not"` dropped accuracy from 93.17% to 87.83% due to false rejections on double negations.
- **v2 Deep Semantic Checks**:
  1. *Numeric Verification*: Checks claim numbers against evidence figures. Permits valid mathematical deductions (e.g. "half of 500" = 250), while flagging true numeric mismatches (e.g. 12% vs 21%).
  2. *Entity Alignment*: Identifies scoped entity substitutions (e.g. `Group X` vs `Group Y`, `Company A` vs `Company B`).
  3. *Causal vs Correlation Gating*: Flags claims asserting causality or proof (`causes`, `proves`) when evidence only establishes correlation (`associated with`, `linked to`).
  4. *Quantifier Scope*: Detects quantifier inflation (`all/every/universally` vs `some/many/majority`).
  5. *Temporal Alignment*: Verifies date and timeframe consistency (e.g. 2022 vs 2023).

### D. 3-Way Entailment Cross-Encoder (`entailment_verifier.py`)
- Evaluates the joint pair `[Evidence Context] [SEP] [Proposition]`.
- Models a full 3-class distribution:
  $$P(\text{entailment}), \quad P(\text{contradiction}), \quad P(\text{insufficient\_evidence})$$
- External Grounding Mapping:
  - $\text{SUPPORTED} \iff P(\text{entailment}) \ge 0.50$
  - $\text{NOT\_SUPPORTED} \iff P(\text{contradiction}) + P(\text{insufficient\_evidence}) > 0.50$
  - If rejected, the internal label distinguishes whether the claim was directly refuted (`CONTRADICTION`) or merely unevidenced (`INSUFFICIENT_EVIDENCE`).

### E. Conservative Decision Aggregation
For any decomposed claim with propositions $\mathcal{P} = \{p_1, \dots, p_m\}$:
$$D(\text{claim}) = \begin{cases} \text{SUPPORTED} & \text{if } \forall p_i \in \mathcal{P}, D(p_i) = \text{SUPPORTED} \\ \text{NOT\_SUPPORTED} & \text{if } \exists p_i \in \mathcal{P}, D(p_i) = \text{NOT\_SUPPORTED} \end{cases}$$
$$P(\text{claim} = \text{SUPPORTED}) = \min_{p_i \in \mathcal{P}} P(p_i = \text{SUPPORTED})$$
This guarantees that high alignment on one trivial clause cannot falsely support a claim containing an ungrounded or contradictory second clause.

---

## 3. Training & Data Synthesis Strategy (`hard_negative_miner_v2.py`)

To resolve the 34 direct-entailment false negatives observed in v1:
1. **Hard Positives**: Synthesize diverse paraphrases, clause reorderings, passive/active voice shifts, and multi-sentence mathematical entailments.
2. **Hard Negatives**: Target near-miss entity, numeric, unit, and causal substitutions.
3. **Loss Function**: Class-balanced cross-entropy + focal loss to penalize overconfidence on easy negatives and force the network to learn subtle entailment syntax.
