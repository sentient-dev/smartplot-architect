"""Application configuration for VastuVision AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "VastuVision AI"
    app_version: str = "0.1.0"
    environment: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    artifacts_dir: str = field(default_factory=lambda: os.getenv("ARTIFACTS_DIR", str(Path.cwd() / "artifacts")))


settings = Settings()
