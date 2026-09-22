# ApexDecision v2: Forensic Architecture Audit Report

**Date**: September 20, 2026  
**Auditor**: Antigravity DeepMind Advanced Agentic Coding  
**Target**: Forensic dissection of v1 ablation identity and evaluation pipeline mechanics  
**Data Artifact**: `apex_decision/benchmarks/grounding_v2/forensic_audit_data.json`  

---

## 1. Executive Summary & Root Cause Confirmation

The v1 Grounding ablation suite reported the following identical results across three ostensibly different configurations:
1. **Ablation 2 (ModernBERT + LoRA)**: 93.17% (41 errors)
2. **Ablation 3 (ModernBERT + LoRA + Hard-Negative Mining)**: 93.17% (41 errors)
3. **Ablation 4 (ModernBERT + LoRA + Evidence Retrieval)**: 93.17% (41 errors)

### Forensic Verdict:
The identity is **100% genuine and mathematically exact at the floating-point level**. The three configurations produced the exact same 600 predictions, with **0 differing predictions**, **0.00000000 mean absolute probability difference**, and identical prediction SHA-256 hashes:
$$\text{Hash: } \texttt{ee64a61b36a57702bf9075ba3b63c74251c418ef0bac0d54ae6ff2b2720fd45b}$$

The root causes are two distinct systemic bottlenecks:
1. **Hard-Negative Mining Saturation Failure**: During v1, `mine_hard_negatives.py` ran inference over `train.json` (Pool A). Because the initial LoRA model had already fit the training set to 100% confidence, zero training samples had $|p - 0.5| < 0.35$ or incorrect labels. Exactly $0$ hard examples were extracted. As a result, `train_augmented.json` was bit-for-bit identical to `train.json`, and `grounding_lora_hn` received identical weights to `grounding_lora`.
2. **Evidence Retrieval Short-Source Pass-Through**: In `test.json`, exactly **600 out of 600 (100.00%)** source texts contain $\le 2$ sentences. The retrieval module's guard clause (`if len(sentences) <= k: return sentences`, where $k=2$) triggered on 100% of test examples. Consequently, the context fed into the encoder was 100% identical to the raw source text, generating identical input tensors and identical forward passes.

---

## 2. Checkpoint and Data File Cryptographic Analysis

| Resource | Path | SHA-256 Checksum | Diff vs Baseline |
| :--- | :--- | :--- | :--- |
| **Base Training Set** | `apex_decision/benchmarks/grounding/train.json` | `19ee3ee801e6ca1dcc592208cbb9ce4c8afff523be3c7231ab41b460d1934f2a` | Baseline |
| **Augmented Training Set** | `apex_decision/benchmarks/grounding/train_augmented.json` | `19ee3ee801e6ca1dcc592208cbb9ce4c8afff523be3c7231ab41b460d1934f2a` | **0 bytes difference (Identical)** |
| **LoRA Adapter Weights** | `apex_decision/build/grounding_lora/adapter_model.safetensors` | `111867160ba793a90623a654bcff6a88b52f90a9b867c2937ea24c3dfcfc80ea` | Baseline |
| **LoRA HN Weights** | `apex_decision/build/grounding_lora_hn/adapter_model.safetensors` | `111867160ba793a90623a654bcff6a88b52f90a9b867c2937ea24c3dfcfc80ea` | **0 bytes difference (Identical)** |

### Answers to Mandatory Hard-Negative Questions:
- **Were hard negatives actually incorporated into training?** No. Exactly 0 examples were mined because `is_error` was False for all 3,400 training samples, and all probabilities exceeded the $0.35$ margin.
- **Did training resume from the expected checkpoint?** Because no new data existed in `train_augmented.json`, retraining with identical hyperparameters and early stopping converged immediately to identical weights.
- **Were the resulting weights different from baseline?** No. The `adapter_model.safetensors` files share the exact SHA-256 hash `111867160ba...`.

---

## 3. Evidence Retrieval Instrumentation Analysis

### Sentence Distribution in Held-Out Test Set ($N=600$):
- **Sources with $\le 2$ sentences**: $600 / 600$ (**100.00%**)
- **Sources with $> 2$ sentences**: $0 / 600$ (**0.00%**)
- **Sources where retrieved context $\ne$ raw source**: $0 / 600$ (**0.00%**)
- **Tokenized input tensor differences (`raw_ids` vs `ret_ids`)**: $0 / 600$ (**0.00%**)

### Code Trace Proof:
In `apex_decision/src/models/grounding_pipeline.py`:
```python
class EvidenceRetriever:
    @classmethod
    def retrieve_top_k(cls, claim: str, sentences: List[str], k: int = 2) -> List[str]:
        if not sentences:
            return []
        if len(sentences) <= k:  # <--- TRIGGERED ON 100% OF TEST CASES
            return sentences
```
When `verify()` executes:
```python
sentences = SentenceSegmenter.segment(source)  # len(sentences) is 1 or 2
evidence_spans = EvidenceRetriever.retrieve_top_k(claim, sentences, k=2)  # returns all sentences
context_text = " ".join(evidence_spans) if evidence_spans else source  # exact reconstruction of source
```
Therefore, in Ablation 4:
$$\text{input\_text} = \texttt{f"\{context\_text\} [SEP] \{claim\}"} \equiv \texttt{f"\{source\} [SEP] \{claim\}"}$$
The token sequences passed to ModernBERT in Ablation 4 were byte-for-byte identical to Ablation 2.

---

## 4. Per-Ablation Empirical Comparison Matrix

| Configuration | Test Set $N$ | Differing Preds vs Abl 2 | Mean Abs $\Delta$ Prob | Max $\Delta$ Prob | Prediction SHA-256 Hash | Trainable Params | Inference Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Frozen ModernBERT** | 600 | 230 | 0.448912 | 0.999812 | `c4b12a8...` | 7,690 | Raw Encoder |
| **2. ModernBERT + LoRA** | 600 | 0 (Base) | 0.000000 | 0.000000 | `ee64a61b...` | 2,305,546 | Raw Encoder |
| **3. LoRA + Hard Negatives**| 600 | 0 | 0.000000 | 0.000000 | `ee64a61b...` | 2,305,546 | Raw Encoder (Identical Weights) |
| **4. LoRA + Retrieval** | 600 | 0 | 0.000000 | 0.000000 | `ee64a61b...` | 2,305,546 | Pipeline (Short-Source Pass-Through)|
| **5. Full Pipeline (+ Rules)**| 600 | 32 | 0.098412 | 0.941200 | `7a92fbc...` | 2,305,546 | Pipeline + Regex Rule Guards |

---

## 5. Architectural Requirements for v2

Based on this forensic audit, the following engineering requirements are mandatory for ApexDecision v2:

1. **Assertion Against Silent Identity**:
   Every ablation and comparative test must compute the SHA-256 hash of predictions and assert $\Delta_{\text{prob}} > 0$ when testing different architectural components on non-trivial datasets.
2. **Substantial Context in Grounding Benchmark (v2)**:
   Source documents must not be limited to 1–2 sentences. Grounding v2 benchmarks must include:
   - Paragraph-length sources (4–10 sentences, 150–400 words).
   - Multi-sentence reasoning dependencies where individual sentences are insufficient or misleading.
   - Distractor sentences containing high lexical overlap with contradictory facts.
3. **True Hard Negative and Hard Positive Generation**:
   Hard negatives cannot be mined solely by querying a model on its own overfitted training split. In v2:
   - Cross-model mining (using out-of-fold predictions or teacher models like NLI-DeBERTa).
   - Generative adversarial mutations (swapping entities, changing numbers, altering temporal ordering, scope inflation).
   - Mining **hard positives** (paraphrases, reordered clauses, multi-sentence evidence) to directly attack the 34 direct-entailment false negatives.
4. **Permanent Removal of Brittle Regex Rule Guards**:
   Ablation 5 proved that naive regex word matching ("never", "no evidence") dropped accuracy from 93.17% to 87.83% due to catastrophic false rejections on double negations and complex syntax. v2 must replace regex guards with deep structured attribute checks and 3-way internal classification (`Entailment`, `Contradiction`, `Insufficient Evidence`).
