import logging
import sys
from pathlib import Path


def setup_logging(workdir: Path) -> None:
    """
    Setup logging to console and file.

    Args:
        workdir: Directory where mlip.log will be created.
    """
    if not workdir.exists():
        workdir.mkdir(parents=True, exist_ok=True)

    log_file = workdir / "mlip.log"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Reset handlers if any
    if logger.handlers:
        logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialised. Log file: {log_file}")
