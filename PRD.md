# Product Requirements Document (PRD)

## Project Name: ApexDecision
Classification: Zero-Backbone Micro-Decision Engine
Hardware Profile: Consumer Laptop (12 GB System RAM, 4 GB dGPU VRAM, 5.9 GB Shared iGPU RAM)
Document Version: 2.0.0
Status: Approved for Implementation

---

## 1. Executive Summary & Problem Definition
Modern production architectures frequently commit hundreds of millions of parameters to simple, deterministic semantic judgments such as policy verification, support ticket routing, safety gating, and sentiment scoring. While proprietary System One services (such as TypeSafe Jev) improve on standard LLMs by replacing autoregressive decoding with prefill evaluations, they still rely on closed cloud infrastructure, network latency (70 ms to 500 ms), and recurring API costs.

ApexDecision is an embedded, zero-backbone decision engine. By completely discarding foundation model weights and utilizing compiled discriminative pipelines (Treelite-compiled gradient-boosted trees and shallow 1D-CNNs), ApexDecision delivers statistically calibrated, schema-guaranteed decisions in under 2.0 ms on local consumer hardware.

---

## 2. Target Personas & Production Use Cases
- Agentic Orchestrator Gating: Autonomous agents needing microsecond authorization checks (policy assertions, tool-call boundaries) before invoking expensive generative reasoning.
- High-Throughput Ingestion Pipelines: Stream processing systems routing tens of thousands of support inquiries, incident reports, or server logs per second.
- Air-Gapped & Embedded Deployments: Environments requiring zero data leakage, zero network connectivity, and minimal runtime memory footprints (< 45 MB).

---

## 3. Native Functional Primitives

### 3.1 BinaryJudge (Assertion Verification)
- Semantics: Evaluates an incoming state string against a binary hypothesis (e.g., is_security_threat, requires_escalation).
- Input Schema:
  - state (string, max 4,000 characters): The raw text context.
  - assertion (string): The boolean assertion to test.
- Output Payload:
  - decision (bool): Mathematical determination (true or false).
  - probability (float): Calibrated epistemic probability p in range [0.0, 1.0].
  - latency_us (int): Execution time in microseconds.

### 3.2 CategoricalJudge (Discrete Multi-Class Routing)
- Semantics: Maps context into exactly one of K mutually exclusive candidate categories (2 <= K <= 64).
- Input Schema:
  - state (string, max 4,000 characters).
  - categories (list of strings): Enumerated choice identifiers.
- Output Payload:
  - decision (string): The chosen category identifier (argmax).
  - probabilities (dictionary of string to float): Normalized probabilities across all K categories.
  - confidence (float): Cardinality-normalized confidence metric: (p_max - 1/K) / (1 - 1/K).
  - latency_us (int): Execution time in microseconds.

### 3.3 OrdinalJudge (Rubric-Based Graded Scoring)
- Semantics: Evaluates context against an ordered set of discrete rubric bins (3 <= M <= 10).
- Input Schema:
  - state (string, max 4,000 characters).
  - rubric_levels (list of strings): Ordered descriptions from lowest to highest grade.
- Output Payload:
  - expected_score (float): Continuous expected value computed across bin mass: sum(m * p_m).
  - bin_probabilities (list of floats): Discrete probability assigned to each grade bin.
  - dispersion (float): Statistical variance across probability distribution.
  - latency_us (int): Execution time in microseconds.

---

## 4. Probability Calibration Standards
Uncalibrated maximum softmax probabilities exhibit severe overconfidence under distribution shifts. ApexDecision mandates calibration:
- Binary Calibration: Platt Scaling fitted via negative log-likelihood on an out-of-fold validation split.
- Categorical Calibration: Temperature Scaling optimizing a strictly positive scalar parameter T via L-BFGS.
- Acceptance Floor: Expected Calibration Error (ECE) across 10 empirical probability bins must satisfy ECE <= 0.035.

---

## 5. Non-Functional Performance Service Level Objectives (SLOs)
- Compiled Tree Latency (p50): <= 0.25 ms (Target), 0.60 ms (Hard Upper Bound)
- Compiled Tree Latency (p99): <= 0.80 ms (Target), 1.50 ms (Hard Upper Bound)
- Neural 1D-CNN Latency (p99): <= 1.80 ms (Target), 3.50 ms (Hard Upper Bound)
- Resident Memory (RSS): <= 30 MB (Target), 50 MB (Hard Upper Bound)
- Single-Core Throughput: >= 4,000 queries/sec (Target), >= 2,500 queries/sec (Floor)
- Deterministic Consistency: 100% bit-level matching across identical inputs.

---

## 6. Failure Modes & Edge-Case Handling
- Out-of-Vocabulary / Unseen Tokens: The hashing vectorizer maps tokens uniformly into fixed hash spaces; unknown tokens never trigger missing-feature exceptions.
- Degenerate / Empty Inputs: Strings with length < 3 characters bypass inference and return a calibrated uniform prior (p = 1/K) with an empty_input_warning flag.
- Input Truncation: States exceeding 4,000 characters are truncated cleanly at token boundaries without memory reallocation or process panics.
