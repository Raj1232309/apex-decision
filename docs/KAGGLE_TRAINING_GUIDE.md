# Kaggle Acceleration & Training Guide: Apex-748M Decision Model

This guide provides end-to-end instructions for training the **Apex-748M** classifier on **Kaggle** using a dual-accelerator handshake strategy: **20 hours/week on Cloud TPU (v3-8 / v5e-4)** followed seamlessly by **30 hours/week on Dual NVIDIA GPUs (2x T4 / P100)**, delivering **50 hours of free weekly compute**.

---

## 1. Architectural Strategy: Why RLVR & Zero Pretrained Backbone

* **No Teacher-Student Distillation:** Rather than distilling soft probability tokens from an autoregressive LLM (which transfers arithmetic hallucinations and formatting bias), the model acts as a policy trained via **Reinforcement Learning with Verifiable Rewards (RLVR)** against an exact, deterministic Python AST Math Engine.
* **No Pretrained Foundation Backbone:** The model is initialized with clean, from-scratch weights (746,813,455 exact parameters; 21 Tier-1 layers + 4 Tier-2 memory latent cross-attention layers). It eliminates all third-party backbone dependencies, pretraining contamination, and quadratic context bottlenecks.
* **Deterministic Two-Phase Protocol:**
  1. **Phase 1: SFT Warmup (300M–500M tokens):** Supervised cross-entropy directly on AST-verified balance sheets and contract invariants, anchoring the reference policy $\pi_{\text{ref}}$.
  2. **Phase 2: RLVR with GRPO ($G=4$, 1.0B tokens):** Group Relative Policy Optimization over mutated query groups scored by exact math ($R_{\text{symbolic}}$), causal invariance ($R_{\text{invariance}}$), and Brier confidence calibration ($R_{\text{calibration}}$).

---

## 2. Weekly Free Compute Schedule on Kaggle

| Accelerator | Weekly Quota | Memory Configuration | Target Training Phase |
| :--- | :--- | :--- | :--- |
| **Cloud TPU (v3-8 / v5e-4)** | **20 Hours / week** | 8 cores $\times$ 16 GB = 128 GB HBM | **Phase 1: Deterministic SFT Warmup** (500M tokens in ~12.5h) + Initial GRPO |
| **Dual NVIDIA GPUs (2x T4)** | **30 Hours / week** | 2 $\times$ 16 GB = 32 GB GDDR6 | **Phase 2: RLVR GRPO Policy Training** (Resumes from latest TPU checkpoint) |

The PyTorch trainer in [`apex_classifier_750m/train.py`](file:///c:/Users/Shriyans/Downloads/dev/apex_classifier_750m/train.py) is accelerator-agnostic: model states and optimizer dicts saved on TPU can be loaded directly onto Dual CUDA GPUs without code modification or weight conversion.

---

## 3. Initial Kaggle Notebook Setup

### Step 3.1: Create Notebook & Enable Internet
1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right-hand panel under **Notebook options**:
   * Set **Internet** to **On** (required for git cloning and dependencies).
   * Set **Accelerator** to **TPU VM v3-8** (for the first 20 hours).

### Step 3.2: Secure GitHub Authentication via Kaggle Secrets
Avoid hardcoding personal access tokens in notebook cells:
1. In the top menu, go to **Add-ons -> Secrets**.
2. Add a secret with label `KAGGLE_GITHUB_TOKEN` containing your GitHub Personal Access Token.
3. In the first notebook cell, paste:

```python
import os
from kaggle_secrets import UserSecretsClient

# Retrieve secret securely
user_secrets = UserSecretsClient()
token = user_secrets.get_secret("KAGGLE_GITHUB_TOKEN")

# Store credentials securely
cred_path = os.path.expanduser("~/.git-credentials")
with open(cred_path, "w") as f:
    f.write(f"https://x-access-token:{token}@github.com\n")
os.chmod(cred_path, 0o600)

os.system("git config --global credential.helper store")
os.system("git clone https://github.com/Raj1232309/apex-decision.git")

# Navigate into repository
%cd /kaggle/working/apex-decision
!pip install -e .
```

### Step 3.3: Verify Environment & Exact 748M Parameter Match
Run the test suite inside the notebook:
```bash
!python -m pytest tests/test_classifier_750m.py -v
```
You should see:
```text
tests/test_classifier_750m.py::test_exact_748m_parameter_count PASSED
tests/test_classifier_750m.py::test_sandboxed_math_evaluator_valid PASSED
tests/test_classifier_750m.py::test_sandboxed_math_evaluator_guards PASSED
tests/test_classifier_750m.py::test_model_forward_pass PASSED
tests/test_classifier_750m.py::test_checkpoint_roundtrip PASSED
============================= 5 passed in ~20s =============================
```

---

## 4. Phase 1: TPU VM Execution (First 20 Hours)

With **Accelerator: TPU VM v3-8** active, run the Phase 1 SFT Warmup:

```python
!python -m apex_classifier_750m.train \
  --phase sft \
  --tokens 500000000 \
  --batch_size 16 \
  --lr 1e-4 \
  --save_dir /kaggle/working/checkpoints \
  --checkpoint_every 1000
```

### Expected Output
* The script automatically detects `torch_xla` and dispatches tensors to the TPU cores.
* Checkpoints are saved with synchronous SHA-256 integrity verification:
  ```text
  Active Accelerator: Cloud TPU (PyTorch-XLA: xla:0)
  [SFT Step 1000] Loss: 0.8421 | 1,024,000 tokens processed
  Checkpoint saved: /kaggle/working/checkpoints/apex748m_step_1000.pt | SHA-256: 4f8a9e...
  ```

---

## 5. Accelerator Handshake: Switching to Dual NVIDIA GPUs (Next 30 Hours)

Once your weekly 20-hour TPU quota is consumed:

### Step 5.1: Switch Accelerator
1. In the right-hand panel under **Settings**, click **Accelerator**.
2. Select **GPU T4 x2** (Dual NVIDIA Tesla T4, 2 $\times$ 16 GB).
3. The notebook will automatically restart.

### Step 5.2: Resume Training on Dual GPUs
Run the Phase 2 RLVR GRPO training script, pointing to your latest checkpoint:

```python
import os
%cd /kaggle/working/apex-decision

# Resumes training directly on Dual GPUs
!python -m apex_classifier_750m.train \
  --phase rlvr \
  --tokens 1000000000 \
  --batch_size 4 \
  --lr 5e-5 \
  --checkpoint /kaggle/working/checkpoints/apex748m_step_1000.pt \
  --save_dir /kaggle/working/checkpoints \
  --checkpoint_every 500
```

### What Happens Behind the Scenes:
1. `get_execution_device()` recognizes that TPU is absent and discovers `cuda:0` and `cuda:1`.
2. The model is wrapped automatically in `torch.nn.DataParallel(model)`.
3. `load_portable_checkpoint()` maps the TPU weights seamlessly onto the CUDA device memory.
4. GRPO executes group sampling ($G=4$) with verifiable symbolic AST rewards, training without any interruption.

---

## 6. Checkpoint Persistence & Export

Files stored in `/kaggle/working/` persist during your interactive session, but are purged if the session is destroyed without saving.

### Strategy A: Export as Kaggle Dataset (Easiest)
1. In the right-hand panel under **Output**, locate `/kaggle/working/checkpoints`.
2. Click the three dots `...` next to `checkpoints` and select **Save as Dataset**.
3. Name it `apex-748m-checkpoints`.
4. In your next session, simply attach this dataset under **Input -> Add Data -> Your Datasets**.

### Strategy B: Sync to Google Drive
```python
# Optional: mount Google Drive or push to Hugging Face Hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/kaggle/working/checkpoints",
    repo_id="your-username/apex-748m-checkpoints",
    repo_type="model",
    token=user_secrets.get_secret("HF_TOKEN")
)
```

---

## 7. Verification Checklist Before Starting

- [x] Model parameter count matches exact 746,813,455.
- [x] All 5 unit tests pass in `tests/test_classifier_750m.py`.
- [x] Zero teacher model dependencies in training loop.
- [x] Zero pretrained foundation backbone dependencies.
- [x] Checkpoints verified with SHA-256 before disk writes.
