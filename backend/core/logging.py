"""Loguru-based logging setup. Import :func:`get_logger` from anywhere."""
from __future__ import annotations

import sys
from functools import lru_cache

from loguru import logger

from backend.core.config import get_settings


@lru_cache(maxsize=1)
def _configure() -> None:
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        "logs/app.log",
        level=settings.log_level,
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
    )


def get_logger(name: str | None = None):
    """Return a configured loguru logger bound with the given name."""
    _configure()
    return logger.bind(module=name or "app")
