# Dataset Card: ApexDecision Scientific Grounding Benchmark

## 1. Benchmark Summary
The **ApexDecision Scientific Grounding Benchmark** is a high-fidelity, leakage-free benchmark designed to evaluate bidirectional transformer models and micro-decision engines on natural language grounding and claim verification.

Unlike generic sentence classification or NLI datasets containing template repetitions, this benchmark enforces strict source-level disjoint partitioning: **zero source documents, premises, or claim templates appearing in the training or calibration splits ever appear in the evaluation splits**.

---

## 2. Dataset Dimensions & Splits

| Split | Number of Examples | SUPPORTED | NOT_SUPPORTED | Adversarial % | Primary Purpose |
|---|---|---|---|---|---|
| **Train** | 3,400 | 1,700 (50.0%) | 1,700 (50.0%) | 73.53% | Model fine-tuning & hard negative mining |
| **Validation** | 600 | 294 (49.0%) | 306 (51.0%) | 73.50% | Hyperparameter selection & early stopping |
| **Calibration** | 600 | 303 (50.5%) | 297 (49.5%) | 73.50% | Exclusively for temperature / Platt calibration |
| **Held-Out Test** | 600 | 304 (50.7%) | 296 (49.3%) | 76.17% | Unseen generalization & metric calculation |
| **Jev-Style Eval** | 108 | 42 (38.9%) | 66 (61.1%) | 50.00% | 108-claim comparison across 6 domains & 4 tiers |
| **Total** | **5,308** | **2,643 (49.8%)** | **2,665 (50.2%)** | **74.06%** | Complete Benchmark Suite |

---

## 3. Domains Represented
1. **Science:** Planetary astrophysics, quantum chromodynamics, molecular biology, condensed matter physics, paleoclimatology.
2. **Medicine:** Clinical cardiology, metabolic therapeutics, oncology (CAR-T), neurology (Alzheimer's), infectious disease prophylaxis.
3. **Finance:** Fintech operations, central banking monetary policy, corporate bond issuances, private equity liquidation, macroeconomic inflation.
4. **Technology:** Relational database replication, distributed stream consensus (KRaft), GPU hardware architectures, memory safety semantics, cloud gateway APIs.
5. **Law / Policy:** Data privacy directives (GDPR), artificial intelligence regulatory frameworks, corporate governance, consumer privacy statutes, anti-bribery compliance.
6. **Business:** Autonomous supply chain logistics, multi-billion-dollar mergers & acquisitions, SaaS ARR telemetry, automotive software recalls, agricultural sourcing.
7. **General Knowledge:** Global waterways, classical historical institutions, geographic physical landmarks, lunar space missions, oceanic bathymetry.
8. **Security:** Upstream software supply chain backdoors (XZ, SolarWinds), mutual TLS cryptographic protocols, injection attack vectors, cloud IAM privilege hierarchies.

---

## 4. The 20 Granular Failure Modes
Every claim is systematically mapped to one of 20 failure modes to enable diagnostic error analysis:

1. `direct_entailment`: Directly entailed by the evidence.
2. `direct_contradiction`: Direct logical contradiction of the evidence.
3. `unsupported_claim`: Claim containing fabricated facts absent from the source.
4. `negation_reversal`: Falsified by adding or removing a negation operator.
5. `double_negation`: Entailed via double negation or litotes.
6. `entity_substitution`: Plausible entity swapped for the true subject.
7. `number_mismatch`: Numerical value or percentage subtly perturbed.
8. `unit_mismatch`: Incompatible metric unit substituted for the true unit.
9. `date_time_mismatch`: Chronological date or epoch altered.
10. `population_mismatch`: Target cohort or subject group substituted.
11. `scope_mismatch`: Localized observation broadened to universal scope.
12. `quantifier_mismatch`: Partial quantifier ("some", "several") elevated to "all" or "none".
13. `causal_overclaim`: Unverified causal mechanism asserted for an outcome.
14. `correlation_vs_causation`: Concurrent observations conflated into a causal relationship.
15. `comparative_claim_mismatch`: Comparative superiority or ranking inverted.
16. `temporal_ordering_error`: Sequence of events reversed or altered.
17. `attribution_mismatch`: Source finding credited to an unrelated entity.
18. `partial_support_claim`: Partially true premise paired with an unsupported extreme clause.
19. `multi_sentence_reasoning`: Entailment requiring multi-clause deductive reasoning.
20. `lexical_overlap_unsupported`: High word overlap designed to fool bag-of-words classifiers while remaining false.

---

## 5. Deduplication & Leakage Prevention Protocol
1. **Source Disjointness:** Sources in Pool A (Train/Val/Calib) share zero overlap with Pool B (Held-Out Test and Jev-Style Suite).
2. **Cryptographic Hashing:** Every `(source, claim)` pair is normalized and hashed via SHA-256 (`hashlib.sha256`).
3. **Cross-Split Verification:** `split.py` asserts that $\text{sources}_{\text{train}} \cap \text{sources}_{\text{test}} = \emptyset$ and $\text{claims}_{\text{train}} \cap \text{claims}_{\text{test}} = \emptyset$.
4. **Calibration Isolation:** Calibration data is partitioned prior to training and evaluated only on untouched test data.
