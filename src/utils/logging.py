"""Logging utilities."""

from __future__ import annotations

import logging

from src.config.settings import settings


def configure_logging() -> None:
    """Configure application logging once."""
    root_logger = logging.getLogger()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
