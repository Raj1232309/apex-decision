"""
Environment and reproducibility validator for ApexDecision External Benchmark Suite.
Records hardware specifications, package versions, cryptographic checksums, and random seeds.
"""

import os
import sys
import json
import random
import platform
import hashlib
import subprocess
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

BENCHMARK_VERSION = "8.0.0"
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED):
    """Sets deterministic random seeds across python, numpy, and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def compute_file_sha256(filepath: str) -> Optional[str]:
    """Computes SHA-256 hash of a file if it exists."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def get_git_commit(cwd: str) -> str:
    """Retrieves current git commit hash if repository is present."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN_NON_GIT_WORKSPACE"


def probe_environment(root_dir: str) -> Dict[str, Any]:
    """Gathers complete environment and hardware specifications."""
    ckpt_path = os.path.join(root_dir, "apex_decision", "models", "apex_student", "scaling_checkpoints", "student-50m-v8.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(root_dir, "apex_decision", "models", "apex_student", "scaling_checkpoints", "student-50m-v77.pt")

    tok_path = os.path.join(root_dir, "apex_decision", "models", "apex_student", "tokenizer_weights", "tokenizer.json")

    env_info = {
        "benchmark_suite_version": BENCHMARK_VERSION,
        "random_seed": DEFAULT_SEED,
        "timestamp_utc": None,  # Populated at run time
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "architecture": platform.architecture()[0]
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version()
        },
        "pytorch": {
            "version": torch.__version__ if torch else "NOT_INSTALLED",
            "cuda_available": torch.cuda.is_available() if torch else False,
            "cuda_version": torch.version.cuda if torch and torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch and torch.cuda.is_available() else None
        },
        "gpu": {
            "device_count": torch.cuda.device_count() if torch and torch.cuda.is_available() else 0,
            "name": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "NO_GPU_DETECTED",
            "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2) if torch and torch.cuda.is_available() else 0.0
        },
        "cpu_and_ram": {
            "cpu_count_logical": os.cpu_count(),
            "cpu_brand": platform.processor(),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2) if psutil else None
        },
        "apex_model": {
            "architecture": "ApexStudent-50M",
            "active_parameters": 46534662,
            "allocated_parameters": 52830000,
            "parameter_ceiling_limit": 200000000,
            "from_scratch": True,
            "pretrained_weights": False,
            "checkpoint_path": os.path.relpath(ckpt_path, root_dir) if os.path.exists(ckpt_path) else None,
            "checkpoint_sha256": compute_file_sha256(ckpt_path) if os.path.exists(ckpt_path) else None,
            "tokenizer_sha256": compute_file_sha256(tok_path) if os.path.exists(tok_path) else None,
            "git_commit": get_git_commit(root_dir)
        }
    }
    return env_info
