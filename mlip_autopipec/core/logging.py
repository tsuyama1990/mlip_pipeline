import logging
from pathlib import Path

from rich.logging import RichHandler

from mlip_autopipec.exceptions import LoggingError


def setup_logging(log_path: Path, level: str = "INFO") -> None:
    """
    Sets up the logging configuration.

    Args:
        log_path: Path to the log file.
        level: Logging level (default: "INFO").

    Raises:
        LoggingError: If logging configuration fails.
    """
    try:
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
    except (OSError, ValueError) as e:
        # Catch specific exceptions: OSError for file issues, ValueError for bad levels
        msg = f"Failed to setup logging: {e}"
        raise LoggingError(msg) from e
