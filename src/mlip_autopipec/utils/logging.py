import logging
import sys
from pathlib import Path


def setup_logging(workdir: Path = Path("."), level: int = logging.INFO) -> None:
    """
    Configure logging for the application.
    Writes logs to console and a file in the working directory.

    Args:
        workdir: The working directory where logs will be stored.
        level: The logging level.
    """
    root = logging.getLogger()

    # Remove existing handlers to avoid duplicates if called multiple times
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    # Ensure workdir exists before creating log file?
    # Usually workdir is created by Orchestrator or user.
    # But logging setup happens early.
    if not workdir.exists():
        workdir.mkdir(parents=True, exist_ok=True)

    log_file = workdir / "mlip.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.info(f"Logging configured. Writing to {log_file}")
