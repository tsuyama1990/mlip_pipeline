import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_path: Path, level: str = "INFO") -> None:
    """
    Sets up the logging configuration.

    Args:
        log_path: Path to the log file.
        level: Logging level (default: "INFO").
    """
    # Create directory for log file if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(log_path, mode="a")
        ],
    )

    # Set logger levels for noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("ase").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Log file: {log_path}")
