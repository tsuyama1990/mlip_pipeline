import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Configures the global logging settings.

    Args:
        level: The logging level (e.g., "DEBUG", "INFO", "WARNING").
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
