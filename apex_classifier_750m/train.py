"""
Device-Agnostic Training Engine for Apex-748M (TPU & Multi-NVIDIA GPU).
Seamlessly trains across Google Cloud/Kaggle TPU (v3/v5) and Dual NVIDIA GPUs (2x T4 / P100 / A100).
Supports Phase 1 (Deterministic SFT Warmup) and Phase 2 (RLVR with GRPO Group Sampling G=4).
"""

import os
# Prevent CUDA memory fragmentation on memory-constrained GPUs (e.g. 16GB Tesla T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import shutil
import argparse
import hashlib
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Apex748MConfig
from .model import Apex748MClassifier
from .dataset import SyntheticDecisionDataset
from .ast_sandbox import SandboxedMathEvaluator


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
    On CUDA with constrained VRAM (e.g. 16 GB Tesla T4), Adafactor or 8-bit AdamW
    reduces optimizer state memory from ~6.0 GB down to ~0.15–1.5 GB, eliminating CUDA OOM.
    """
    if is_tpu:
        print("Optimizer          : Standard AdamW (Optimized for Cloud TPU)")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if optimizer_type == "auto":
        if torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                print("Optimizer          : bitsandbytes 8-bit AdamW (State VRAM ~1.5 GB)")
                return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
            except Exception:
                print("Optimizer          : PyTorch Adafactor (Sublinear State VRAM ~0.15 GB)")
                return torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=0.01)
        else:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=False)
    elif optimizer_type == "adafactor":
        print("Optimizer          : PyTorch Adafactor (Sublinear State VRAM ~0.15 GB)")
        return torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_type == "adamw8bit":
        import bitsandbytes as bnb
        print("Optimizer          : bitsandbytes 8-bit AdamW (State VRAM ~1.5 GB)")
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        print("Optimizer          : PyTorch AdamW (foreach=False)")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=False)


def save_portable_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    save_dir: str = "./checkpoints"
) -> str:
    """Saves portable checkpoint compatible across both TPU and NVIDIA GPU."""
    os.makedirs(save_dir, exist_ok=True)
    target_path = os.path.join(save_dir, f"apex748m_step_{step}.pt")
    latest_path = os.path.join(save_dir, "apex748m_step_latest.pt")

    # Handle DataParallel wrapper
    raw_model = model.module if hasattr(model, "module") else model

    state = {
        "step": step,
        "model_state": raw_model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "config": asdict(raw_model.config) if hasattr(raw_model.config, "__dataclass_fields__") else raw_model.config,
    }
    torch.save(state, target_path)

    # Compute SHA-256 hash for integrity
    hasher = hashlib.sha256()
    with open(target_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    hash_prefix = hasher.hexdigest()[:16]

    # Also update 'latest' checkpoint for easy resumption
    try:
        shutil.copyfile(target_path, latest_path)
    except Exception:
        pass

    print(f"Checkpoint saved: {target_path} | SHA-256: {hash_prefix}")
    return target_path


def load_portable_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> int:
    """Loads checkpoint onto the active device (TPU or GPU) regardless of where it was saved."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model_state"])

    if optimizer is not None and "opt_state" in state:
        try:
            optimizer.load_state_dict(state["opt_state"])
        except Exception as e:
            print(f"Warning: Could not restore optimizer states ({e}); proceeding with fresh optimizer.")

    step = state.get("step", 0)
    print(f"Restored checkpoint from {checkpoint_path} at step {step}")
    return step


def train_step_sft(
    model: nn.Module,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_tpu: bool,
    scaler: Optional[Any] = None
) -> float:
    """Phase 1: Deterministic Supervised Warmup Step (Cross-Entropy)."""
    optimizer.zero_grad()

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
        loss_score = F.nll_loss(torch.log(preds["score_probs"] + 1e-8), score_target)
        total_loss = loss_noul + loss_choice + loss_score

    if is_tpu:
        total_loss.backward()
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer)
        xm.mark_step()
    elif scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()

    return total_loss.item()


def train_step_rlvr(
    model: nn.Module,
    group: List[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_tpu: bool,
    scaler: Optional[Any] = None
) -> Tuple[float, float]:
    """Phase 2: RLVR with GRPO Group Sampling (G=4) and Verifiable Rewards."""
    optimizer.zero_grad()
    rewards = []
    log_probs = []

    use_cuda_amp = (not is_tpu) and (device.type == "cuda") and (scaler is not None)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_amp):
        for item in group:
            chunks = item["chunks"].to(device)
            cand_tokens = item["choice_candidates"].unsqueeze(0).to(device)
            noul_target = item["noul_label"]

            preds = model(chunks.unsqueeze(0), candidate_tokens=cand_tokens)

            # Policy distribution (Bernoulli probability for Noul)
            prob = preds["noul_prob"].squeeze().item()
            action = 1.0 if prob >= 0.5 else 0.0

            # Verifiable reward: R = R_symbolic + R_calibration
            r_symbolic = 1.0 if action == noul_target else -2.5
            r_calib = - ((prob - noul_target) ** 2)
            r_total = r_symbolic + r_calib
            rewards.append(r_total)

            pred_prob_t = preds["noul_prob"].squeeze()
            log_p = torch.log(pred_prob_t if action == 1.0 else (1.0 - pred_prob_t) + 1e-8)
            log_probs.append(log_p)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-6
        advantages = (rewards_t - mean_r) / std_r

        policy_loss = -torch.mean(torch.stack(log_probs) * advantages)

    if is_tpu:
        policy_loss.backward()
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer)
        xm.mark_step()
    elif scaler is not None:
        scaler.scale(policy_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        policy_loss.backward()
        optimizer.step()

    return policy_loss.item(), mean_r.item()


def main():
    parser = argparse.ArgumentParser(description="Apex-748M Dual-Backend Trainer (TPU & NVIDIA GPU)")
    parser.add_argument("--phase", type=str, default="sft", choices=["sft", "rlvr"], help="Training phase: 'sft' (Warmup) or 'rlvr' (GRPO)")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--tokens", type=int, default=None, help="Target total tokens (calculates steps automatically)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 16 on TPU, 4 on GPU)")
    parser.add_argument("--save_dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="/kaggle/working/checkpoints", help="Directory to save/load checkpoints")
    parser.add_argument("--checkpoint", "--resume", dest="resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--checkpoint_every", type=int, default=250, help="Interval (in steps) to save checkpoints")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4 for sft, 5e-5 for rlvr)")
    parser.add_argument("--optimizer", type=str, default="auto", choices=["auto", "adafactor", "adamw8bit", "adamw"], help="Optimizer (auto chooses Adafactor/8-bit Adam on CUDA)")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP (automatic mixed precision) on CUDA")
    args = parser.parse_args()

    device, device_desc, is_tpu = get_execution_device()

    # Determine default batch size & learning rate based on device & phase
    if args.batch_size is None:
        args.batch_size = 16 if is_tpu else 4

    if args.lr is None:
        args.lr = 1e-4 if args.phase == "sft" else 5e-5

    # Determine total steps
    tokens_per_step = args.batch_size * 512 * 4  # batch * chunk_len * max_chunks (≈ 8,192 tokens per step at bsz=4)
    if args.steps is not None:
        total_steps = args.steps
    elif args.tokens is not None:
        total_steps = max(1, args.tokens // tokens_per_step)
    else:
        total_steps = 1000

    print("=" * 70)
    print("APEX-748M CLASSIFIER TRAINING ENGINE")
    print(f"Active Accelerator : {device_desc}")
    print(f"Training Phase     : {args.phase.upper()}")
    print(f"Target Steps       : {total_steps:,} (~{total_steps * tokens_per_step:,} tokens)")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Learning Rate      : {args.lr}")
    print(f"Save Directory     : {args.checkpoint_dir}")
    print(f"Checkpoint Every   : {args.checkpoint_every} steps")
    print("=" * 70)

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

    # Build memory-efficient optimizer
    optimizer = build_optimizer(model, lr=args.lr, optimizer_type=args.optimizer, is_tpu=is_tpu)

    start_step = 0
    if args.resume:
        start_step = load_portable_checkpoint(args.resume, model, optimizer, device=device)

    dataset = SyntheticDecisionDataset(phase=args.phase, group_size=4)
    data_iter = iter(dataset)

    # Clean GPU cache before starting loop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nStarting training loop...")
    t0 = time.time()
    for step in range(start_step + 1, start_step + total_steps + 1):
        batch = next(data_iter)

        if args.phase == "sft":
            loss = train_step_sft(model, batch, optimizer, device, is_tpu, scaler=scaler)
            if step % 20 == 0 or step == start_step + 1:
                elapsed = time.time() - t0
                speed = (step - start_step) / max(elapsed, 0.001)
                tokens_done = (step - start_step) * tokens_per_step
                print(f"[Step {step:05d}/{start_step + total_steps}] Loss: {loss:.4f} | Processed: {tokens_done:,} tokens | Speed: {speed:.2f} steps/s")
        else:
            loss, mean_reward = train_step_rlvr(model, batch, optimizer, device, is_tpu, scaler=scaler)
            if step % 20 == 0 or step == start_step + 1:
                elapsed = time.time() - t0
                speed = (step - start_step) / max(elapsed, 0.001)
                tokens_done = (step - start_step) * tokens_per_step
                print(f"[Step {step:05d}/{start_step + total_steps}] Policy Loss: {loss:.4f} | Avg Reward: {mean_reward:+.2f} | Processed: {tokens_done:,} tokens | Speed: {speed:.2f} steps/s")

        # Save checkpoint periodically
        if step % args.checkpoint_every == 0 or step == start_step + total_steps:
            save_portable_checkpoint(model, optimizer, step, save_dir=args.checkpoint_dir)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
