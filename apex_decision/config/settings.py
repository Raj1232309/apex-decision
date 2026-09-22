"""Environment-based configuration with safe local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    device: str
    host: str
    port: int
    api_key: str | None
    max_tool_calls: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        package_dir = Path(__file__).resolve().parents[1]
        default_model = package_dir / "models" / "apex_student" / "scaling_checkpoints" / "student-50m-v8.pt"
        return cls(
            model_path=Path(os.getenv("APEX_MODEL_PATH", default_model)),
            device=os.getenv("APEX_DEVICE", "auto"),
            host=os.getenv("APEX_HOST", "127.0.0.1"),
            port=int(os.getenv("APEX_PORT", "8000")),
            api_key=os.getenv("APEX_API_KEY") or None,
            max_tool_calls=int(os.getenv("APEX_MAX_TOOL_CALLS", "5")),
            log_level=os.getenv("APEX_LOG_LEVEL", "INFO"),
        )
