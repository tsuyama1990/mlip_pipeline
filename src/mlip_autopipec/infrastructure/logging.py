import logging

from rich.logging import RichHandler

from mlip_autopipec.domain_models.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Configure the root logger with Rich console output and file logging.

    Args:
        config: Logging configuration object.
    """
    # Create the handlers
    handlers: list[logging.Handler] = []

    # Rich Handler for console
    rich_handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
    handlers.append(rich_handler)

    # File Handler
    file_handler = logging.FileHandler(config.file_path)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # Configure Root Logger
    logging.basicConfig(
        level=config.level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
