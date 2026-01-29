import logging
from pathlib import Path

from rich.logging import RichHandler


def configure_logging(level: str = "INFO", log_file: Path | str | None = None) -> None:
    """
    Configure the root logger with Rich console output and file logging.

    Args:
        level: Logging level for the console (default: INFO).
        log_file: Path to the log file. If provided, file logging is enabled at DEBUG level.
    """
    # Create handlers
    handlers: list[logging.Handler] = []

    # Rich Handler for console
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(level)
    handlers.append(console_handler)

    # File Handler
    root_level = level
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        # If file logging is enabled (DEBUG), root must be DEBUG to allow messages to reach file handler
        root_level = "DEBUG"

    # Configure Root Logger
    logging.basicConfig(
        level=root_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True
    )
