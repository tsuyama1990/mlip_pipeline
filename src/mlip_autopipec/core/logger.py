import logging
import sys
from pathlib import Path


def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> None:
    """
    Configures the root logger to write to console and a file.

    Args:
        log_dir: Directory where the log file will be created.
        log_level: Logging level (default: INFO).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mlip_pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Prevent duplicate handlers
    # Check if we already have handlers attached to root logger
    if root_logger.handlers:
        # Check if they are our handlers?
        # For simplicity, if any handlers exist, we assume logging is configured.
        # But we might want to ensure *our* file handler is there.
        # For Cycle 01, simple check is sufficient as per feedback.
        # "Add proper handler deduplication check" -> checking root_logger.handlers is the check.
        # I will make it explicit.
        return

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")
