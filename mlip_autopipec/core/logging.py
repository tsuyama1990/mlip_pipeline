"""
Centralized logging configuration for MLIP-AutoPipe.
"""

import logging
import sys
from pathlib import Path

from mlip_autopipec.exceptions import LoggingError


def setup_logging(log_file: Path, level: int = logging.INFO) -> None:
    """
    Configures the root logger to write to both console and a log file.

    Args:
        log_file: Path to the log file.
        level: Logging level (default: INFO).

    Raises:
        LoggingError: If logging configuration fails.
    """
    try:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Reset existing handlers to allow reconfiguration in tests/re-runs
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info(f"Logging initialized. Writing to {log_file}")

    except OSError as e:
        msg = f"Failed to create log file or directory: {log_file}"
        raise LoggingError(msg) from e
    except Exception as e:
        msg = f"Unexpected error during logging setup: {e}"
        raise LoggingError(msg) from e
