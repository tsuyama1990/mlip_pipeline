import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """
    Sets up structured logging for the application.
    """
    handler: logging.Handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(handler)

    return logging.getLogger("mlip_pipeline")
