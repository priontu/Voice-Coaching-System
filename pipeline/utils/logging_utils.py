"""
utils/logging_utils.py - Structured logging setup for VocalCoach.

Named logging_utils.py (not logging.py) to avoid shadowing the stdlib
logging module on any Python path configuration.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DEFAULT_DATE_FMT = "%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> None:
    """
    Configure the root logger for VocalCoach.

    Call once at application startup (e.g. in an inference entry point).
    Subsequent calls update the root handler's level.

    Args:
        level:   Logging level string: DEBUG, INFO, WARNING, ERROR.
        fmt:     Log format string. Defaults to the VocalCoach standard.
        datefmt: Date format string.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format=fmt or _DEFAULT_FORMAT,
        datefmt=datefmt or _DEFAULT_DATE_FMT,
        stream=sys.stderr,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)
