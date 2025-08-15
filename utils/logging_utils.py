"""Centralized, JSON-formatted logging utilities (singleton configuration).

- Configure a single root StreamHandler (JSON) exactly once.
- Avoid duplicate handlers and double-logging.
- Provide write_status() that attributes to the caller (stacklevel=2).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

__all__ = ["configure_logging", "get_logger", "write_status", "set_level"]

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Minimal JSON line formatter: timestamp, level, module, message."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        # timestamp in ISO 8601 with UTC 'Z'
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S,%f"
        )[:-3]
        payload: Dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "module": record.name or record.module,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single JSON StreamHandler on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers to stop duplicate lines.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Prevent libraries from adding noisy extra handlers through propagation.
    logging.captureWarnings(True)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger without adding handlers (uses root’s single handler)."""
    return logging.getLogger(name if name else __name__)


def write_status(msg: str, level: int = logging.INFO) -> None:
    """Log a status line, attributed to the **caller** (not this helper)."""
    logger = logging.getLogger(__name__)
    # stacklevel=2 attributes module/line to the immediate caller if Python>=3.8
    try:
        logger.log(level, msg, stacklevel=2)
    except TypeError:
        # Fallback for very old Python: attribute to this module
        logger.log(level, msg)


def set_level(level: int) -> None:
    """Dynamically raise/lower root log level."""
    logging.getLogger().setLevel(level)
