import logging
import sys
from pathlib import Path


def setup_logging(work_dir: Path, level: int = logging.INFO) -> None:
    """
    Sets up logging for the application.
    Logs to console (level) and to file (DEBUG).
    """
    log_file = work_dir / "orchestrator.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture everything

    # Check if handlers already exist to prevent duplication
    if logger.handlers:
        return

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")
