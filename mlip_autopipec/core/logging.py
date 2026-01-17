import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_path: Path, level: str | int = logging.INFO) -> None:
    """
    Sets up the logging configuration.

    Args:
        log_path: Path to the log file.
        level: Logging level (default: logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )
