import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the logging system with a standard format and stdout handler.

    Args:
        level: The logging level (default: logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Ensure we override any existing config
    )
