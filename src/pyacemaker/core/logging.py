"""Centralized logging configuration."""

import sys
from typing import Any

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure the logging system.

    Args:
        level: The logging level (e.g., "DEBUG", "INFO").

    """
    logger.remove()
    logger.configure(extra={"name": "pyacemaker"})
    logger.add(sys.stderr, level=level, format="[{time}] [{level}] [{extra[name]}] {message}")


def get_logger(name: str) -> Any:
    """Get a logger instance with the specified name.

    Args:
        name: The name of the logger (usually __name__ or class name).

    Returns:
        A logger instance bound to the given name.

    """
    return logger.bind(name=name)
