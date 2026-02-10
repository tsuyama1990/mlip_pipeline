import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: str = "mlip_pipeline.log") -> None:
    """
    Configures the centralized logger with Rich console output and file logging.
    Prevents duplicate handlers if called multiple times.
    """
    logger = logging.getLogger("mlip_autopipec")

    # Prevent duplicate handlers (compatibility with pytest)
    if logger.handlers:
        return

    logger.setLevel(level)

    # Console Handler (Rich)
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
