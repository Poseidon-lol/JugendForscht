"""
Logging helpers for the OSC discovery project.
===============================================

We rely on Python's built-in :mod:`logging` infrastructure but provide a
slightly opinionated configuration utility:

* Colorised console output (when supported)
* Optional file logging with rotating handlers
* Consistent formatter shared across notebooks, CLIs, and scripts

Usage
-----
>>> from src.utils.log import setup_logging, get_logger
>>> setup_logging(log_file='experiments/run_001/train.log')
>>> logger = get_logger(__name__)
>>> logger.info("Training started")
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

__all__ = ["setup_logging", "get_logger"]


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if _is_tty() and record.levelno in self.COLORS:
            return f"{self.COLORS[record.levelno]}{message}{self.RESET}"
        return message


def _is_tty() -> bool:
    stream = getattr(sys, "stderr", None)
    return hasattr(stream, "isatty") and stream.isatty()


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    *,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configure root logger and optional file handler."""

    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(level)
    # clear existing handlers (safe for multiple invocations)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = _ColorFormatter(formatter._fmt, formatter.datefmt)  # type: ignore[arg-type]

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Shortcut to obtain module logger."""

    return logging.getLogger(name)
