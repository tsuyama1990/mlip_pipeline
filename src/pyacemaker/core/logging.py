"""Centralized logging configuration."""

import sys
from typing import Any

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure the logging system.

    Args:
        level: The logging level (e.g., "DEBUG", "INFO"). Must be one of:
               DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Raises:
        ValueError: If the log level is invalid.

    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level.upper() not in valid_levels:
        msg = f"Invalid log level: {level}. Must be one of {valid_levels}"
        raise ValueError(msg)

    logger.remove()
    logger.configure(extra={"name": "pyacemaker"})
    logger.add(sys.stderr, level=level, format="[{time}] [{level}] [{extra[name]}] {message}")


def get_logger(name: str) -> Any:  # Loguru type stubs are incomplete
    """Get a logger instance with the specified name.

    Args:
        name: The name of the logger (usually __name__ or class name).

    Returns:
        A logger instance bound to the given name.

    """
    return logger.bind(name=name)
