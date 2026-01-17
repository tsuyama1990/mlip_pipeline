import logging
import sys
from pathlib import Path
from rich.logging import RichHandler

def setup_logging(log_path: Path) -> None:
    """
    Sets up the centralized logging system.
    Configures a StreamHandler (Rich) for console and a FileHandler for the log file.
    """
    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler (Rich)
    console_handler = RichHandler(rich_tracebacks=True, show_time=True, show_level=True)
    console_handler.setLevel(logging.INFO)

    # File Handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.DEBUG) # Log everything to file
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Log file: {log_path}")
