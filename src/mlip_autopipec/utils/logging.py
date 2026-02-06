import logging
import sys
from pathlib import Path


def setup_logging(log_file: Path = Path("mlip_pipeline.log"), level: int = logging.INFO) -> None:
    """
    Configures the root logger to log to stdout and a file.

    Args:
        log_file: Path to the log file.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")
