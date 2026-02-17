"""Logging utilities."""

from __future__ import annotations

import logging

from src.config.settings import settings


def configure_logging() -> None:
    """Configure application logging once."""
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
