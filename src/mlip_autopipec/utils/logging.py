import logging
import sys


def setup_logging(level: int | str = logging.INFO) -> None:
    """
    Configures the root logger with a standard format.

    Args:
        level: Logging level (int or str, e.g., "DEBUG").
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Remove existing handlers to avoid duplicates on re-config
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override existing handlers (Python 3.8+)
    )
