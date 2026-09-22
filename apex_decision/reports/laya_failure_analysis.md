# Laya Failure Analysis & Architectural Bottleneck Report

**Date:** September 21, 2026  
**Subject System:** Laya System 1 Decision Engine (`laya` v0.3.4)  
**Evaluated Artifacts:** `laya/common.py`, `laya/agent.py`, `laya/lang.py`, `research/BENCHMARKS.md`, `research/results/`  

---

## 1. Executive Summary

Laya represents a significant advancement in non-autoregressive decision models, demonstrating that structured classification can be executed in a single forward pass under 35 ms without language model generation artifacts. 

However, deep source-code auditing and empirical benchmarking reveal **five fundamental architectural bottlenecks** in Laya. These failure modes explain why Laya fails on large option spaces, cannot handle enterprise document contexts, collapses out-of-domain without warning, and lacks verifiable evidence provenance.

---

## 2. Failure Mode 1: Hard Context Truncation (512 / 1024 Tokens)

### 2.1 The Code Mechanism
In `laya/common.py` (lines 80–86):
```python
room = max(0, max_len - len(ids) - 1)
st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
st = st[-room:] if truncate_left else st[:room]
ids = ids + st + [tok.sep_token_id]
return ids[:max_len], [m for m in markers if m < max_len]
```

### 2.2 Empirical Impact
1. **Total Context Blindness**: In an enterprise document of 5,000 to 256,000 tokens (contracts, financial reports, technical manuals), Laya evaluates only the first 300–450 tokens. Any critical evidence, clause, amendment, or number appearing beyond token 512 is **completely dropped**.
2. **False Negatives & Hallucinated Abstention**: Because the model is physically prevented from seeing 99%+ of the document, questions whose answers lie deeper in the text receive near-uniform or default predictions.
3. **No Dynamic Needle Retrieval**: Laya has no hierarchical index, no memory state, and no chunking mechanism.

---

## 3. Failure Mode 2: High-Cardinality Option Collapse (Banking77 & Large Choices)

### 3.1 The Code Mechanism
In `laya/common.py` (lines 70–75):
```python
opt_budget = head_max_len - sum(len(o) for o in opt_ids)
if opt_budget < 16:
    per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
    opt_ids = [o[:per] for o in opt_ids]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
head_ids = head_ids[: max(8, opt_budget)]
```

### 3.2 Empirical Impact
1. **Subword Truncation Disaster**: When a question presents 77 options (such as the Banking77 benchmark), `opt_ids` are allocated `per = max(4, (192 - 16) // 77) = 4` tokens per option!
2. **Loss of Semantic Distinguishability**: An option like `"card_arrival_estimate_inquiry"` is truncated to `["card", "_", "arrival", "_"]`. All sub-intent qualifiers are severed.
3. **Severe Accuracy Drop**: As documented in Laya's own `BENCHMARKS.md`:
   - Banking77: Laya achieves **0.425 accuracy**, compared to TypeSafe Jev's published **0.870**.
   - As stated by the Laya authors: *"Banking77 is the one clear loss, and it is architectural: a choice question's options share a fixed head_max_len budget, so 77 labels get roughly 4 tokens each and stop being distinguishable."*
4. **Failure on >100 and 255 Choices**: Options beyond ~40 are either severely truncated or pushed beyond the sequence boundary, making high-cardinality routing impossible.

---

## 4. Failure Mode 3: Silent Out-of-Domain Overconfidence

### 4.1 The Mechanism
Laya's confidence is computed solely from the Shannon entropy of the output probability distribution:
$$C(p) = 1 - \frac{H(p)}{\ln(K)}$$
The model has no epistemic absence detection mechanism. When presented with completely unfamiliar text, out-of-vocabulary alphabets, or unseen domain structures, the attention weights often collapse onto a single arbitrary option, resulting in a low-entropy (and therefore high-confidence) incorrect prediction.

### 4.2 Empirical Findings in Laya's Own Benchmarks
- On the Khmer language (51-language sweep), the English checkpoint scored **0.000 accuracy at 0.952 mean confidence**.
- The macro ECE across 51 languages for the English model was **0.7331**, with mean confidence never dropping below 0.885 at any accuracy level.
- Confidence gating completely fails to catch errors because the model remains supremely confident while completely wrong.

---

## 5. Failure Mode 4: Lack of Exact Evidence Grounding & Provenance

### 5.1 The Mechanism
Laya operates strictly at the sequence classification level. It returns:
```json
{
  "department": {
    "type": "choice",
    "choice": "billing",
    "confidence": 0.94,
    "probabilities": {"billing": 0.94, "technical": 0.03, ...}
  }
}
```
There is no token pointer, no start/end offset extraction, and no evidence verification.

### 5.2 Enterprise Risk
In high-stakes enterprise workflows (legal compliance, clinical trial verification, fraud detection, cybersecurity triage), an assertion without exact character-level provenance cannot be audited by human reviewers. If a model claims a claim is `SUPPORTED` but cannot cite the exact sentence, the decision cannot be trusted.

---

## 6. Failure Mode 5: Fine-Grained Numerical and Polarity Reversal

### 6.1 The Mechanism
Because bidirectional cross-attention without explicit numeric reasoning tokens treats numbers similarly to standard subwords, subtle numerical relationships and comparative polarities are easily confused:
- Comparative reductions: $-2.30\%$ vs $-1.86\%$.
- Negation inversions: *"Treatment showed no adverse events"* vs *"Treatment showed zero efficacy"*.
- Temporal supersessions: An amendment stating *"Section 4.1 is hereby superseded and replaced by Section 4.2"* requires distinguishing the obsolete clause from the active binding clause.

Laya's dense marker head has no symbolic or arithmetic grounding, leading to false-positive entailments when key entities match but numeric polarities invert.

---

## 7. Architectural Directives for ApexStudent

To transcend Laya's limitations, ApexStudent implements:
1. **Hierarchical 256K–1M Context Streaming**: Ingests arbitrary document lengths via chunked block encoding, local-global salience routing, and recurrent memory tokens.
2. **2-Stage Hierarchical Choice Head**: Efficiently handles up to 255 options without subword truncation via coarse semantic clustering followed by fine-grained marker ranking.
3. **Calibrated Epistemic Abstention**: Integrates explicit absence detection and Brier-regularized abstention heads.
4. **Deterministic Provenance Verification**: Locates exact character offsets and verifies substring fidelity before certifying an output.
5. **Symbolic & Numeric Polarity Guards**: Enforces deterministic mathematical and logical consistency across percentages, inequalities, and dates.
