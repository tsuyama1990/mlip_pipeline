"""
Centralized logging configuration for MLIP-AutoPipe.
"""
import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """
    Sets up the logging configuration using RichHandler.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    # Suppress loud libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
