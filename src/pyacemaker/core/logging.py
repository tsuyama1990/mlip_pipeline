"""Centralized logging configuration."""

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

from pyacemaker.core.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Configure the logging system.

    Args:
        config: Logging configuration object.

    """
    logger.remove()
    logger.configure(extra={"name": "pyacemaker"})
    logger.add(
        sys.stderr,
        level=config.level,
        format=config.format,
    )


def get_logger(name: str) -> "Logger":
    """Get a logger instance with the specified name.

    Args:
        name: The name of the logger (usually __name__ or class name).

    Returns:
        A logger instance bound to the given name.

    """
    return logger.bind(name=name)
