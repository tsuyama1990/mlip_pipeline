import logging
import sys


def setup_logging(
    level: int | str = logging.INFO,
    format_str: str | None = None,
) -> None:
    """
    Configures the root logger with a standard format.

    Args:
        level: Logging level (int or str, e.g., "DEBUG").
        format_str: Optional custom format string.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Remove existing handlers to avoid duplicates on re-config
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override existing handlers (Python 3.8+)
    )
