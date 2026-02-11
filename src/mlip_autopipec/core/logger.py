import logging
from pathlib import Path


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure the root logger for the application.
    Ensures handlers are not duplicated.
    """
    logger = logging.getLogger("mlip_autopipec")
    logger.setLevel(level)

    # Prevent duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
