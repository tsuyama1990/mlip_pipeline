import logging
import sys


def setup_logging(name: str = "mlip_autopipec", level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure the logging system for the application.

    Args:
        name: The name of the logger to configure.
        level: The logging level (e.g., "DEBUG", "INFO", "WARNING").
        log_file: Optional path to a file where logs should be written.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Name of the logger, typically __name__.

    Returns:
        logging.Logger: The configured logger.
    """
    return logging.getLogger(name)
