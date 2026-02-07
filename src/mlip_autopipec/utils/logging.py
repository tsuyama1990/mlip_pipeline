import logging

from rich.logging import RichHandler


def configure_logging(
    level: str = "INFO", log_file: str = "mlip_pipeline.log", force: bool = False
) -> None:
    """
    Configures the root logger to output to console (Rich) and file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers if force is True
    if force:
        root_logger.handlers = []

    if root_logger.hasHandlers():
        return

    # Console Handler (Rich)
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root_logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
