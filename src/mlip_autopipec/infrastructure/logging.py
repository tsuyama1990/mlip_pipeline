"""Logging configuration using Rich and standard logging."""

import logging
from pathlib import Path

from rich.logging import RichHandler

from mlip_autopipec.constants import DATE_FORMAT, DEFAULT_LOG_FILENAME, LOG_FORMAT


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    verbose: bool = False,
) -> None:
    """Configure the root logger with Rich handler and File handler.

    Args:
        log_level: The logging level for the console (default: INFO).
        log_file: Path to the log file. If None, uses DEFAULT_LOG_FILENAME.
        verbose: If True, sets console level to DEBUG.
    """
    if verbose:
        log_level = "DEBUG"

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture everything at root level

    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Console Handler (Rich)
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_time=True,
        show_level=True,
    )
    console_handler.setLevel(level)
    # RichHandler formats internally, so we don't set a formatter for it
    logger.addHandler(console_handler)

    # File Handler
    if log_file is None:
        log_file = Path(DEFAULT_LOG_FILENAME)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Suppress noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
