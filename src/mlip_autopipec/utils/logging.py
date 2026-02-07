import logging
import sys


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure the global logger.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("mlip_autopipec").setLevel(level)
