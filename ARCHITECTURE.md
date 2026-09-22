# Architectural Specification: ApexDecision

## 1. High-Level Architectural Pipeline
ApexDecision completely bypasses autoregressive transformers, sequence packing, and attention matrices. The system relies on a two-path discriminative architecture:

- Path A (Ultra-Low-Latency Tree Path): Subword and character n-gram feature hashing coupled with an ensemble of gradient-boosted decision trees compiled to branchless native C via Treelite. Optimized for CPU execution in under 0.5 ms.
- Path B (Deep Semantic Neural Path): Character and word embeddings passed through parallel 1D convolutional filter banks (TextCNN) exported to static ONNX graphs. Executed via ONNX Runtime on CPU or the 4 GB discrete GPU in 1.0 ms to 2.0 ms.

---

## 2. Ingestion & Featurization Pipeline

### 2.1 Fast MurmurHash3 Sparse Vectorizer
Traditional language models rely on multi-megabyte vocabulary lookup tables that incur cache misses and memory synchronization overhead. ApexDecision utilizes vocabulary-free feature hashing:
- N-Gram Extractors: Extracts word unigrams, word bigrams, and character 3-grams simultaneously.
- Hash Dimension: Fixed sparse vector length of N = 32,768 buckets.
- Hash Kernel: MurmurHash3 32-bit integer variant mapping string slices directly to column indices: feature_idx = MurmurHash3(token) mod N.
- Value Normalization: Sublinear Term-Frequency scaling combined with L2 norm: x_i = (1 + log(count_i)) / norm(x).
- Memory Cost: Zero dynamic allocation. The vectorizer populates a pre-allocated sparse Compressed Sparse Row (CSR) structure directly.

---

## 3. Engine Deep Dives

### 3.1 Path A: Compiled Tree Engine (Treelite)
- Base Estimator: Histogram-based Gradient Boosted Trees (150 trees, max depth 8, 63 max leaves, learning rate 0.08).
- Compilation Pipeline:
  1. Python model is trained on labeled domain samples.
  2. Model trees are translated into a Treelite C abstract syntax tree.
  3. Treelite compiles the ensemble into optimized native C source code where tree traversals are unrolled into flat, branch-predicted conditionals.
  4. GCC, Clang, or MSVC produces a platform-native shared object library (libapex_tree.so or apex_tree.dll).
  5. The runtime binds to the shared library via ctypes/C-FFI, evaluating inference passes entirely within C native memory.

### 3.2 Path B: Shallow 1D-CNN (ONNX Runtime)
For tasks where semantic phrase ordering is critical:
- Architecture: TextCNN (Kim, 2014) trained from scratch without pre-trained embeddings.
  - Embedding Layer: Vocabulary capacity of 10,000, embedding dimension D = 128.
  - Convolutional Banks: Three parallel 1D convolutions with kernel sizes 2, 3, and 4 (64 filters each, 192 filters total).
  - Pooling: Global 1D temporal max-over-time pooling extracting the most prominent feature per kernel.
  - Projection: Fully connected layer with 20% dropout mapping to decision classes.
- Export Pipeline:
  - Model graph is frozen and exported to ONNX (opset 17) with fixed input tensor shapes (1 x 256).
  - Served using onnxruntime.InferenceSession configured with GraphOptimizationLevel.ORT_ENABLE_ALL and multi-threaded CPU/CUDA execution providers.

---

## 4. Probability Calibration Layer
Raw model margins z do not represent true empirical probabilities. ApexDecision integrates explicit post-hoc calibration:

### 4.1 Platt Calibration (Binary Primitives)
Learns a scalar multiplier A and bias B by minimizing cross-entropy loss over a validation set:
P(y = 1 | z) = 1 / (1 + exp(A * z + B))

### 4.2 Temperature Calibration (Categorical & Ordinal Primitives)
Optimizes a single temperature parameter T > 0 using L-BFGS:
P(y = i | z) = exp(z_i / T) / sum(exp(z_j / T))
This preserves the classification argmax while compressing or expanding the probability distribution to match empirical accuracy.

---

## 5. Thread Safety & Memory Model
- Zero Allocations per Call: All intermediate memory buffers (sparse pointers, input tensors, output probability arrays) are pre-allocated during process startup.
- Re-Entrant Execution: Both the compiled Treelite C library and the ONNX Runtime sessions are stateless and thread-safe, supporting parallel inference across multi-threaded ASGI workers without global locks.
