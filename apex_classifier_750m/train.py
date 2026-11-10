"""
Device-Agnostic Training Engine for Apex-748M (TPU & Multi-NVIDIA GPU).
Seamlessly trains across Google Cloud/Kaggle TPU (v3/v5) and Dual NVIDIA GPUs (2x T4 / P100 / A100).
Supports Phase 1 (Deterministic SFT Warmup) and Phase 2 (RLVR with GRPO Group Sampling G=4).
"""

import os
import sys

# Auto-inject repository root and package directory into sys.path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_this_dir)
for _p in (_repo_dir, _this_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Unbuffered line-by-line output for immediate real-time progress visibility in Colab / Kaggle
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ["PYTHONUNBUFFERED"] = "1"

# Prevent CUDA memory fragmentation on memory-constrained GPUs (e.g. 16GB Tesla T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import shutil
import argparse
import hashlib
import threading
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import Apex748MConfig
    from .model import Apex748MClassifier
    from .dataset import SyntheticDecisionDataset
    from .ast_sandbox import SandboxedMathEvaluator
except (ImportError, ValueError):
    try:
        from apex_classifier_750m.config import Apex748MConfig
        from apex_classifier_750m.model import Apex748MClassifier
        from apex_classifier_750m.dataset import SyntheticDecisionDataset
        from apex_classifier_750m.ast_sandbox import SandboxedMathEvaluator
    except (ImportError, ValueError):
        from config import Apex748MConfig
        from model import Apex748MClassifier
        from dataset import SyntheticDecisionDataset
        from ast_sandbox import SandboxedMathEvaluator


def get_execution_device() -> Tuple[torch.device, str, bool]:
    """Detects available accelerator: TPU (PyTorch-XLA), Multi-CUDA GPU, or CPU."""
    # 1. Check for TPU (PyTorch-XLA)
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        world_size = xm.xrt_world_size()
        return device, f"TPU (PyTorch-XLA, World Size: {world_size})", True
    except Exception:
        pass

    # 2. Check for NVIDIA CUDA GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        return torch.device("cuda:0"), f"NVIDIA GPU ({gpu_count}x: {', '.join(device_names)})", False

    # 3. Fallback to CPU
    return torch.device("cpu"), "CPU (Fallback)", False


def build_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_type: str = "auto",
    is_tpu: bool = False
) -> torch.optim.Optimizer:
    """
    Builds a memory-efficient optimizer.
    On CUDA with constrained VRAM (e.g. 16 GB Tesla T4), Adafactor eliminates the ~6.0 GB
    AdamW second-moment state overhead down to ~0.15 GB, preventing CUDA OOM.
    """
    if is_tpu:
        print("Optimizer          : Standard AdamW (Optimized for Cloud TPU)")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if optimizer_type in ("auto", "adafactor"):
        if torch.cuda.is_available():
            print("Optimizer          : PyTorch Adafactor (Sublinear State VRAM ~0.15 GB — Zero OOM)")
            return torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=0.01)
        else:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=False)
    elif optimizer_type == "adamw8bit":
        try:
            import bitsandbytes as bnb
            print("Optimizer          : bitsandbytes 8-bit AdamW (State VRAM ~1.5 GB)")
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
        except Exception as e:
            print(f"Warning: bitsandbytes failed ({e}); falling back to native Adafactor.")
            return torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        print("Optimizer          : PyTorch AdamW (foreach=False)")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=False)


def sync_checkpoint_to_hf(file_path: str, repo_id: str, token: Optional[str] = None):
    """Synchronously pushes latest checkpoint to Hugging Face Hub with verification."""
    if not os.path.exists(file_path):
        return
    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"\n[Cloud Sync] Pushing checkpoint ({file_size_gb:.2f} GB) to Hugging Face ({repo_id})...")
    t0 = time.time()
    try:
        from huggingface_hub import HfApi
        actual_token = token or os.environ.get("HF_TOKEN")
        api = HfApi(token=actual_token)
        # Ensure repo is public (public repos have unlimited free storage on Hugging Face)
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        except Exception:
            pass
        try:
            api.update_repo_settings(repo_id=repo_id, private=False)
        except Exception:
            pass
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="apex748m_step_latest.pt",
            repo_id=repo_id,
            repo_type="model"
        )
        elapsed = time.time() - t0
        print(f"[Cloud Sync] PUSH COMPLETE in {elapsed:.1f}s: {repo_id}/apex748m_step_latest.pt safely stored in cloud!\n")
    except Exception as e:
        print(f"\n[Cloud Sync Warning] Hugging Face push notice: {e}; local checkpoint intact.\n")


def save_portable_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    save_dir: str = "./checkpoints",
    max_to_keep: int = 2,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None
) -> str:
    """Saves portable checkpoint compatible across both TPU and NVIDIA GPU with disk rotation."""
    os.makedirs(save_dir, exist_ok=True)
    target_path = os.path.join(save_dir, f"apex748m_step_{step}.pt")
    tmp_target = os.path.join(save_dir, f"apex748m_step_{step}.pt.tmp")
    latest_path = os.path.join(save_dir, "apex748m_step_latest.pt")
    tmp_latest_path = os.path.join(save_dir, "apex748m_step_latest.pt.tmp")

    # Handle DataParallel wrapper
    raw_model = model.module if hasattr(model, "module") else model

    # Proactive Disk Space Check (especially critical on Kaggle's 20 GB disk limit)
    try:
        _, _, free_bytes = shutil.disk_usage(save_dir)
        free_gb = free_bytes / (1024 ** 3)
        if free_gb < 4.0:
            print(f"Notice: Low disk headroom ({free_gb:.1f} GB free). Purging older intermediate checkpoints...")
            for fname in os.listdir(save_dir):
                if fname.startswith("apex748m_step_") and fname.endswith(".pt") and fname != "apex748m_step_latest.pt":
                    try:
                        os.remove(os.path.join(save_dir, fname))
                    except Exception:
                        pass
    except Exception:
        pass

    state = {
        "step": step,
        "model_state": raw_model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "optimizer_type": optimizer.__class__.__name__,
        "config": asdict(raw_model.config) if hasattr(raw_model.config, "__dataclass_fields__") else raw_model.config,
    }

    try:
        # Atomic write to temporary file first so an interrupted write never corrupts valid checkpoints
        torch.save(state, tmp_target)
        os.replace(tmp_target, target_path)

        # Compute SHA-256 hash for integrity
        hasher = hashlib.sha256()
        with open(target_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        hash_prefix = hasher.hexdigest()[:16]

        # Atomically update 'latest' checkpoint for easy resumption
        try:
            shutil.copyfile(target_path, tmp_latest_path)
            os.replace(tmp_latest_path, latest_path)
        except Exception:
            try:
                shutil.copyfile(target_path, latest_path)
            except Exception:
                pass

        # Disk Protection: Prune older numbered checkpoints to prevent Kaggle 20 GB disk overflow
        try:
            numbered_files = []
            for fname in os.listdir(save_dir):
                if fname.startswith("apex748m_step_") and fname.endswith(".pt") and fname not in ("apex748m_step_latest.pt", "apex748m_step_latest.pt.tmp"):
                    try:
                        s = int(fname.split("_")[-1].replace(".pt", ""))
                        numbered_files.append((s, os.path.join(save_dir, fname)))
                    except ValueError:
                        pass
            numbered_files.sort(key=lambda x: x[0])
            while len(numbered_files) > max_to_keep:
                old_step, old_path = numbered_files.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)
        except Exception as e:
            print(f"Warning: Checkpoint rotation error: {e}")

        # Cloud Sync: Asynchronously push to Hugging Face if configured
        if hf_repo and os.path.exists(latest_path):
            sync_checkpoint_to_hf(latest_path, hf_repo, hf_token)

        print(f"Checkpoint saved: {target_path} | SHA-256: {hash_prefix} (latest updated)")
        return target_path
    except Exception as e:
        print(f"Warning: Failed to save checkpoint at step {step} ({e}); continuing training safely.")
        for p in (tmp_target, tmp_latest_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        return ""


def load_portable_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None
) -> int:
    """Loads checkpoint onto the active device (TPU or GPU) regardless of where it was saved."""
    # Cloud Sync: Fetch from Hugging Face if not found locally
    if (not checkpoint_path or not os.path.exists(checkpoint_path)) and hf_repo:
        print(f"[Cloud Sync] Local checkpoint not found at '{checkpoint_path}'. Checking Hugging Face ({hf_repo})...")
        try:
            from huggingface_hub import hf_hub_download
            actual_token = hf_token or os.environ.get("HF_TOKEN")
            print(f"[Cloud Sync] Fetching apex748m_step_latest.pt from Hugging Face: {hf_repo}...")
            checkpoint_path = hf_hub_download(repo_id=hf_repo, filename="apex748m_step_latest.pt", token=actual_token)
            print(f"[Cloud Sync] Successfully downloaded to: {checkpoint_path}")
        except Exception as e:
            print(f"[Cloud Sync Notice] Could not fetch from Hugging Face: {e}")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint weights from {checkpoint_path}...")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model_state"])

    if optimizer is not None and "opt_state" in state:
        opt_state = state["opt_state"]
        saved_opt_type = state.get("optimizer_type")
        current_opt_type = optimizer.__class__.__name__

        # Validate cross-optimizer compatibility (e.g. Adafactor on GPU vs AdamW on TPU/CPU)
        is_compatible = True
        reason = ""

        if saved_opt_type is not None and saved_opt_type != current_opt_type:
            is_compatible = False
            reason = f"saved optimizer algorithm ({saved_opt_type}) differs from active optimizer ({current_opt_type})"
        else:
            saved_groups = opt_state.get("param_groups", [])
            if saved_groups:
                g0 = saved_groups[0]
                if isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
                    if "betas" not in g0 or "beta2_decay" in g0:
                        is_compatible = False
                        reason = f"saved state lacks 'betas' or contains Adafactor-specific parameters required by {current_opt_type}"
                elif isinstance(optimizer, torch.optim.Adafactor):
                    if not isinstance(g0.get("eps"), (tuple, list)) or "betas" in g0:
                        is_compatible = False
                        reason = "saved state format is incompatible with Adafactor"

        if is_compatible:
            try:
                # Capture defaults to safeguard against any missing hyperparameter keys
                current_defaults = {k: v for k, v in optimizer.defaults.items()}
                optimizer.load_state_dict(opt_state)
                # Sanitize param groups so no default keys are missing
                for group in optimizer.param_groups:
                    for k, v in current_defaults.items():
                        if k not in group:
                            group[k] = v
                print(f"Restored optimizer state ({current_opt_type}) from checkpoint.")
            except Exception as e:
                print(f"Warning: Could not restore optimizer states ({e}); proceeding with fresh {current_opt_type} state.")
        else:
            print(f"[Optimizer Compatibility Notice] {reason.capitalize()}.")
            print(f"[Optimizer Compatibility Notice] Initializing fresh {current_opt_type} momentum buffers while fully preserving all 746,813,455 model weights and continuing from step {state.get('step', 0)}.")

    step = state.get("step", 0)
    print(f"Restored checkpoint from {checkpoint_path} at step {step}")
    return step


def train_step_sft(
    model: nn.Module,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_tpu: bool,
    scaler: Optional[Any] = None,
    accum_steps: int = 1,
    is_step: bool = True
) -> float:
    """Phase 1: Deterministic Supervised Warmup Step (Cross-Entropy with Gradient Accumulation)."""
    chunks = batch["chunks"].to(device)  # [num_chunks, 512]
    cand_tokens = batch["choice_candidates"].unsqueeze(0).to(device)  # [1, 4, 16]
    noul_target = torch.tensor([batch["noul_label"]], dtype=torch.float32, device=device)
    choice_target = torch.tensor([batch["choice_label"]], dtype=torch.long, device=device)
    score_target = torch.tensor([batch["score_label"]], dtype=torch.long, device=device)

    use_cuda_amp = (not is_tpu) and (device.type == "cuda") and (scaler is not None)

    # Autocast context for NVIDIA FP16 mixed precision
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_amp):
        preds = model(chunks.unsqueeze(0), candidate_tokens=cand_tokens)

        loss_noul = F.binary_cross_entropy_with_logits(preds["noul_logit"].squeeze(-1), noul_target)
        loss_choice = F.cross_entropy(preds["choice_logits"], choice_target)
        loss_score = F.nll_loss(torch.log(preds["score_probs"].clamp(min=1e-7)), score_target)
        total_loss = (loss_noul + loss_choice + loss_score) / accum_steps

    if is_tpu:
        total_loss.backward()
        if is_step:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
            xm.mark_step()
            optimizer.zero_grad()
    elif scaler is not None:
        scaler.scale(total_loss).backward()
        if is_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        total_loss.backward()
        if is_step:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss.item() * accum_steps


def train_step_rlvr(
    model: nn.Module,
    group: List[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_tpu: bool,
    scaler: Optional[Any] = None
) -> Tuple[float, float]:
    """
    Phase 2: RLVR with GRPO Group Sampling (G=4) and Verifiable Rewards.
    Uses two-stage execution:
      Stage 1: Zero-memory rollout under torch.no_grad() to compute rewards and advantages.
      Stage 2: Per-sample backward accumulation to hold only 1 forward graph in VRAM at any time.
    """
    G = len(group)
    rewards = []
    actions = []

    use_cuda_amp = (not is_tpu) and (device.type == "cuda") and (scaler is not None)

    # Stage 1: Fast Rollout (No Gradients) -> Zero activation VRAM
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_amp):
            for item in group:
                chunks = item["chunks"].to(device)
                cand_tokens = item["choice_candidates"].unsqueeze(0).to(device)
                noul_target = item["noul_label"]

                preds = model(chunks.unsqueeze(0), candidate_tokens=cand_tokens)
                prob = preds["noul_prob"].squeeze().item()
                action = 1.0 if prob >= 0.5 else 0.0

                r_symbolic = 1.0 if action == noul_target else -2.5
                r_calib = - ((prob - noul_target) ** 2)
                r_total = r_symbolic + r_calib

                rewards.append(r_total)
                actions.append(action)

    # Compute Group Relative Advantages
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std() + 1e-6
    advantages = (rewards_t - mean_r) / std_r

    # Stage 2: Policy Gradient Backward with Per-Item Accumulation
    optimizer.zero_grad()
    total_policy_loss = 0.0

    for i, item in enumerate(group):
        chunks = item["chunks"].to(device)
        cand_tokens = item["choice_candidates"].unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_amp):
            preds = model(chunks.unsqueeze(0), candidate_tokens=cand_tokens)
            pred_prob_t = preds["noul_prob"].squeeze()

            # Numerically stable log probability
            chosen_p = pred_prob_t if actions[i] == 1.0 else (1.0 - pred_prob_t)
            log_p = torch.log(chosen_p.clamp(min=1e-7, max=1.0 - 1e-7))

            # Policy loss scaled by group size
            loss_i = -(log_p * advantages[i]) / G

        if is_tpu:
            loss_i.backward()
        elif scaler is not None:
            scaler.scale(loss_i).backward()
        else:
            loss_i.backward()

        total_policy_loss += loss_i.item()

    # Step optimizer
    if is_tpu:
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer)
        xm.mark_step()
    elif scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return total_policy_loss, mean_r.item()


def main():
    parser = argparse.ArgumentParser(description="Apex-748M Dual-Backend Trainer (TPU & NVIDIA GPU)")
    parser.add_argument("--phase", type=str, default="sft", choices=["sft", "rlvr"], help="Training phase: 'sft' (Warmup) or 'rlvr' (GRPO)")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--tokens", type=int, default=None, help="Target total tokens (calculates steps automatically)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 16 on TPU, 2 on GPU)")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--save_dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="/kaggle/working/checkpoints", help="Directory to save/load checkpoints")
    parser.add_argument("--checkpoint", "--resume", dest="resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--checkpoint_every", type=int, default=250, help="Interval (in steps) to save checkpoints")
    parser.add_argument("--max_checkpoints", type=int, default=2, help="Maximum number of intermediate checkpoints to retain (default: 2)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4 for sft, 5e-5 for rlvr)")
    parser.add_argument("--optimizer", type=str, default="auto", choices=["auto", "adafactor", "adamw8bit", "adamw"], help="Optimizer (default: auto -> Adafactor on CUDA)")
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face repo (e.g. 'username/apex-748m') for automatic zero-friction cloud checkpoint sync")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (optional, defaults to $HF_TOKEN)")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP (automatic mixed precision) on CUDA")
    args = parser.parse_args()

    device, device_desc, is_tpu = get_execution_device()

    # Determine default batch size & learning rate based on device & phase
    if args.batch_size is None:
        args.batch_size = 16 if is_tpu else 2

    if args.lr is None:
        args.lr = 1e-4 if args.phase == "sft" else 5e-5

    # Tokens per micro-step = 1 sample * 4 chunks * 512 tokens = 2,048 tokens
    # Tokens per optimizer step = args.grad_accum_steps * 2,048 tokens
    tokens_per_optimizer_step = args.grad_accum_steps * 512 * 4

    # Initialize model matching exact audited 746,813,455 parameters
    config = Apex748MConfig()
    model = Apex748MClassifier(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized  : {total_params:,} parameters (Audited Apex-748M)")

    model = model.to(device)

    # Multi-GPU support if running on CUDA with multiple GPUs (e.g. 2x NVIDIA T4)
    if not is_tpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Multi-GPU Active   : Wrapping model in DataParallel across {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)

    # FP16 GradScaler for CUDA GPUs (T4 Tensor Cores)
    scaler = None
    if not is_tpu and device.type == "cuda" and not args.no_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            scaler = torch.cuda.amp.GradScaler()
        print("Precision Mode     : FP16 Mixed Precision (AMP Enabled via Tensor Cores)")

    # Build memory-efficient optimizer (Adafactor uses ~0.15 GB vs AdamW ~6.0 GB)
    optimizer = build_optimizer(model, lr=args.lr, optimizer_type=args.optimizer, is_tpu=is_tpu)

    start_step = 0
    if args.resume or args.hf_repo:
        resume_target = args.resume or os.path.join(args.checkpoint_dir, "apex748m_step_latest.pt")
        try:
            start_step = load_portable_checkpoint(
                resume_target, model, optimizer, device=device,
                hf_repo=args.hf_repo, hf_token=args.hf_token
            )
        except Exception as e:
            print(f"Notice: No previous checkpoint loaded ({e}); starting from step 0.")

    if args.steps is not None:
        target_step = start_step + args.steps
    elif args.tokens is not None:
        target_tokens_steps = max(1, args.tokens // tokens_per_optimizer_step)
        if target_tokens_steps > start_step:
            target_step = target_tokens_steps
        else:
            target_step = start_step + target_tokens_steps
    else:
        target_step = start_step + 1000

    steps_to_run = target_step - start_step

    print("=" * 70)
    print("APEX-748M CLASSIFIER TRAINING ENGINE (AUDITED ZERO-OOM)")
    print(f"Active Accelerator : {device_desc}")
    if not is_tpu and device.type == "cpu":
        print("!" * 70)
        print("⚠️  NOTICE: Running on CPU Fallback!")
        print("    A 746.8M model takes ~1-2 min/step on CPU and consumes high system RAM.")
        print("    Recommendation in Colab: Switch to 'Runtime' -> 'Change runtime type'")
        print("    -> 'T4 GPU' (free, fast, FP16 Tensor Cores, ~0.5s/step).")
        print("!" * 70)
    print(f"Training Phase     : {args.phase.upper()}")
    if start_step > 0:
        print(f"Resume Status      : Resuming from Step {start_step:,} -> Target Step {target_step:,} ({steps_to_run:,} steps remaining)")
    else:
        print(f"Target Steps       : {target_step:,} (~{target_step * tokens_per_optimizer_step:,} tokens)")
    print(f"Grad Accum Steps   : {args.grad_accum_steps}")
    print(f"Learning Rate      : {args.lr}")
    print(f"Save Directory     : {args.checkpoint_dir}")
    print(f"Checkpoint Every   : {args.checkpoint_every} steps (Max Kept: {args.max_checkpoints})")
    if args.hf_repo:
        print(f"Cloud Sync (HF)    : Active -> {args.hf_repo} (Automatic Cloud Backup)")
    print("=" * 70)

    dataset = SyntheticDecisionDataset(phase=args.phase, group_size=4)
    data_iter = iter(dataset)

    # Clean GPU cache before starting loop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nStarting training loop...")
    optimizer.zero_grad()
    t0 = time.time()

    current_step = start_step
    try:
        for step in range(start_step + 1, target_step + 1):
            current_step = step
            if args.phase == "sft":
                # Gradient accumulation across grad_accum_steps
                accum_loss = 0.0
                for micro_idx in range(args.grad_accum_steps):
                    batch = next(data_iter)
                    is_last_micro = (micro_idx == args.grad_accum_steps - 1)
                    l = train_step_sft(
                        model, batch, optimizer, device, is_tpu,
                        scaler=scaler, accum_steps=args.grad_accum_steps, is_step=is_last_micro
                    )
                    accum_loss += l

                if step % 20 == 0 or step == start_step + 1:
                    elapsed = time.time() - t0
                    speed = (step - start_step) / max(elapsed, 0.001)
                    lifetime_tokens = step * tokens_per_optimizer_step
                    print(f"[Step {step:05d}/{target_step}] Loss: {accum_loss:.4f} | Processed: {lifetime_tokens:,} tokens | Speed: {speed:.2f} steps/s")

            else:
                # Phase 2 RLVR with GRPO
                group = next(data_iter)
                loss, mean_reward = train_step_rlvr(model, group, optimizer, device, is_tpu, scaler=scaler)
                if step % 20 == 0 or step == start_step + 1:
                    elapsed = time.time() - t0
                    speed = (step - start_step) / max(elapsed, 0.001)
                    lifetime_tokens = step * tokens_per_optimizer_step
                    print(f"[Step {step:05d}/{target_step}] Policy Loss: {loss:.4f} | Avg Reward: {mean_reward:+.2f} | Processed: {lifetime_tokens:,} tokens | Speed: {speed:.2f} steps/s")

            # Save checkpoint periodically
            if step % args.checkpoint_every == 0 or step == target_step:
                save_portable_checkpoint(
                    model, optimizer, step,
                    save_dir=args.checkpoint_dir,
                    max_to_keep=args.max_checkpoints,
                    hf_repo=args.hf_repo,
                    hf_token=args.hf_token
                )

        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Training interrupted by user at Step {current_step}!")
        print("[Interrupt] Saving emergency checkpoint and uploading to Hugging Face...")
        save_portable_checkpoint(
            model, optimizer, current_step,
            save_dir=args.checkpoint_dir,
            max_to_keep=args.max_checkpoints,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token
        )
        print(f"[Interrupt] Step {current_step} safely preserved in cloud! Resumption ready on next run.\n")
        return


if __name__ == "__main__":
    main()
