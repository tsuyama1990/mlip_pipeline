"""
Centralized logging configuration for MLIP-AutoPipe.
"""

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """
    Configures the root logger to write to console (Rich) and optional log file.

    Args:
        log_file: Path to the log file.
        level: Logging level (default: INFO).
    """
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
    ]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
