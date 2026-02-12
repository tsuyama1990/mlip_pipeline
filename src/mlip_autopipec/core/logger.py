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
    if root_logger.handlers:
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
