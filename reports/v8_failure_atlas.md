# ApexDecision V8: Permanent Failure Taxonomy & Baseline Failure Atlas

**Dataset Evaluated**: `evaluation/v76_ood_dataset.json` (10,010 Cases, SHA-256: `A54534FAFFB9CBA6AADD23CDCD346F3B534624D31034261480D9678046102DB6`)  
**Architecture**: `ApexStudent-50M` (FROZEN: 46.53M Active Parameters, 100% From Scratch)  
**Baseline Model Checkpoint**: `student-50m-v77.pt` (SHA-256: `04317B9CD695CD70BC1BEB201A017ADF2640376406C18D58CD0F332A5318E15B`)  
**Total Evaluated Cases**: 10,010  
**Baseline Passing Cases**: 9,768 (97.58%)  
**Total Baseline Failures**: **242 (2.42%)**  
**JSON Catalog**: [`reports/v8_failure_atlas.json`](file:///c:/Users/Shriyans/Downloads/dev/reports/v8_failure_atlas.json)  

---

## 1. Permanent Failure Taxonomy Specification

Every failure in ApexDecision is classified under the multi-axial hierarchical taxonomy:

### Axis 1: Semantic Reasoning
- `negation`: Inversion of assertion polarity ("not", "never", "unapproved").
- `double_negation`: Multiple negations cancelling or shifting truth ("cannot fail to include").
- `scope`: Universal vs existential quantification ("all servers" vs "at least one server").
- `implication`: Consequence entailment without explicit syntactic equivalence.
- `contradiction`: Incompatible claims across propositions ("zero downtime" vs "3 hr outage").
- `exception`: Specific exclusionary carve-outs ("except for scheduled maintenance").
- `temporal_ordering`: Timestamp and version precedence ("v2.1 supersedes v2.0").
- `comparative_reasoning`: Greater-than, less-than, equal-to relations.
- `causal_confusion`: Confusing correlation with causal prerequisite.
- `instruction_conflict`: Contradictions between system guidelines and user prompts.
- `criteria_conflict`: Mutually exclusive option criteria in choice tasks.
- `entity_confusion`: Conflating distinct entities sharing attributes or context.
- `coreference`: Pronoun or alias misresolution across sentences.
- `compositional_reasoning`: Multi-clause synthesis requiring multiple deduction steps.

### Axis 2: Noul (Belief Grounding)
- `false_positive`: Predicting supported when evidence is insufficient or contradictory.
- `false_negative`: Predicting ungrounded when evidence strictly entails the claim.
- `uncertainty_miscalibration`: Extreme overconfidence or underconfidence near decision boundaries.
- `evidence_sufficiency_error`: Failing to discern passing mention from conclusive proof.
- `abstention_error`: Failing to abstain when evidence is absent or ungrounded.
- `subtle_semantic_polarity`: Subtle framing shifts altering truth status.
- `tool_needed_detection_error`: Misclassifying a task requiring external data retrieval.

### Axis 3: Choice
- `high_cardinality_options`: Performance degradation as option count $K \rightarrow 1,000$.
- `distractor_confusion`: Selecting near-miss distractors over the exact match.
- `semantically_equivalent_options`: Ambiguity between redundant options.
- `near_duplicate_options`: Fine-grained discrimination errors.
- `option_omission`: Failing to recognize when the correct option is missing.
- `hierarchical_selection_failure`: Routing errors in coarse-to-fine multi-level menus.

### Axis 4: Score
- `ordinal_confusion`: Inverting rank order in score scales.
- `rubric_boundary_confusion`: Threshold rounding errors on metric boundaries.
- `interpolation_error`: Inability to map continuous values into discrete tiers.
- `inconsistent_rubric_application`: Applying different standards across instances.

### Axis 5: Arithmetic
- `carry_borrow`: Basic multi-digit arithmetic alignment.
- `fractions_decimals_percentages`: Ratio, basis-point, and decimal computation.
- `chained_operations`: Sequential operations where output $N$ feeds input $N+1$.
- `long_dependency_chain`: 5-hop, 10-hop, 20-hop multi-step calculations.
- `irrelevant_number_distractors`: Numbers appearing in text not part of calculation.
- `symbolic_numeric_mixture`: Currency, units, and scientific notation parsing.

### Axis 6: Multilingual
- `script_coverage`: Latin, Cyrillic, Arabic, Devanagari, Bengali, Tamil, Telugu, CJK, Thai, Hebrew.
- `mixed_script_code_switching`: Interleaved English and native language tokens.
- `transliterated_text`: Native words written in Latin phonetics.
- `multilingual_options_evidence`: Prompt in Language A, evidence in Language B.

### Axis 7: Long Context
- `context_depth`: 32K, 64K, 128K, 256K, 512K, 1M tokens.
- `distributed_evidence`: Premises separated by thousands of distractor tokens.
- `distractor_saturation`: High-density irrelevant numerical and entity noise.
- `cross_document_contradiction`: Reconciling assertions from conflicting documents.

### Axis 8: Tool Agent
- `tool_needed_classification`: Accurately detecting when external tool is required.
- `no_tool_classification`: Avoiding unnecessary tool invocations for self-contained tasks.
- `tool_selection`: Selecting the precise tool from the registry.
- `argument_extraction`: Mapping user context to JSON schema properties.
- `argument_validation`: Strict typing, enum, range, and format enforcement.
- `permission_violation`: Attempting side-effecting operations without authorization.
- `stale_contradictory_results`: Handling invalid, stale, or conflicting tool outputs.
- `tool_loop_cycle`: Infinite call loops and duplicate invocations.
- `tool_hallucination`: Claiming tool output without verified execution log.
- `result_verification`: Verifying whether tool return satisfies the goal.

---

## 2. Baseline Failure Cluster Distribution (The 242 V7.7 Failures)

The 242 failures in `student-50m-v77.pt` on the 10,010 OOD suite are distributed as follows:

```
+------------------------------------+---------------+------------------------------------------------------+
| Failure Category                   | Count         | Primary Taxonomy Tags                                |
+------------------------------------+---------------+------------------------------------------------------+
| v76_multilingual:German            | 15            | multilingual, inflectional_morphology, compound_word |
| v76_multilingual:Chinese           | 15            | multilingual, non_latin_script, cjk_segmentation     |
| v76_multilingual:Japanese          | 15            | multilingual, non_latin_script, cjk_segmentation     |
| v76_multilingual:Arabic            | 15            | multilingual, non_latin_script, right_to_left        |
| v76_multilingual:Russian           | 15            | multilingual, non_latin_script, cyrillic             |
| v76_multilingual:Hindi             | 15            | multilingual, non_latin_script, devanagari           |
| v76_multilingual:Bengali           | 15            | multilingual, non_latin_script, bengali              |
| v76_multilingual:Tamil             | 15            | multilingual, non_latin_script, tamil                |
| v76_multilingual:Telugu            | 15            | multilingual, non_latin_script, telugu               |
| v76_multihop:10_hops               | 51            | arithmetic, long_dependency_chain, chained_ops       |
| v76_multihop:3_hops                | 6             | arithmetic, multi_hop, dependency_chain              |
| v76_noul:entity_attribution        | 50            | noul, entity_confusion, distractor_confusion         |
+------------------------------------+---------------+------------------------------------------------------+
| TOTAL BASELINE FAILURES            | 242           |                                                      |
+------------------------------------+---------------+------------------------------------------------------+
```

### Summary of Root Causes:
1. **Multilingual (135 cases)**:
   The existing subword tokenizer vocabulary (`vocab_size=4096`) was trained predominantly on English tokens. Non-Latin characters (Devanagari, Bengali, Cyrillic, Arabic, CJK, Tamil, Telugu) were falling back to byte modulo indexing, which stripped syntactic distinctions.
2. **Extreme Multi-Hop Arithmetic (57 cases)**:
   In 10-hop dependency chains with 10 intermediate calculations, token truncation and attention dilution across long state documents caused the neural student to lose precision on late-stage sums.
3. **Noul Entity Attribution (50 cases)**:
   In complex financial/legal documents with multiple similar entity names (e.g., "Subsidiary A" vs "Subsidiary B"), the model occasionally associated a valid predicate with the wrong subject.

---

## 3. V8 Engineering Resolution Plan

1. **Genuine Multilingual Pipeline**: Unicode NFKC normalization + script-aware encoding that preserves character identities and exact evidence span provenance across all 100+ languages.
2. **Structured Deterministic Arithmetic Engine**: An operation planner extracting entities, values, operations, and executing deterministic computations with verified proof chains up to 20 hops.
3. **Noul Hard-Negative Minimal-Pair Verifier**: Subject-predicate alignment checking preventing entity attribution bleed.
4. **Native Tool Orchestrator**: Deterministic execution substrate for typed tool calling, argument validation, and loop prevention.
