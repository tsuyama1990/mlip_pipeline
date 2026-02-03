import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configures the logging for the application."""

    # Create formatters
    # Detailed format for file
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Simpler format for console (can be colored if needed)
    console_fmt = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)

    # Silence external libraries if they are too noisy (optional)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pydantic").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger for the given module name."""
    return logging.getLogger(name)
