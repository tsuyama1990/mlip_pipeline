import logging
import sys
from pathlib import Path

from mlip_autopipec.config import validate_safe_path


def setup_logging(
    name: str = "mlip_pipeline",
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure structured logging for the application.
    """
    if log_file:
        validate_safe_path(log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicates
    # But if log_file is provided and not present in handlers, we might want to add it
    # For now, simplistic check to avoid double-add
    if logger.handlers:
        # Check if we need to add the new file handler?
        # Ideally we shouldn't reuse the same logger name for different configs in the same process
        # without cleanup.
        # But let's assume we proceed if handlers exist.
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def shutdown_logging() -> None:
    """
    Close all logging handlers to ensure file handles are released.
    Useful for cleanup and tests.
    """
    # We iterate over all loggers? Or just the root?
    # logging.shutdown() closes everything.
    logging.shutdown()
