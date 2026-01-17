import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_path: Path) -> None:
    """
    Sets up the logging configuration.

    Args:
        log_path: Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )
