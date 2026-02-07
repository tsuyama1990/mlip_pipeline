import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure the global logging settings.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        msg = f"Invalid log level: {log_level}"
        raise ValueError(msg)  # noqa: TRY004

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Reconfigure if already configured
    )
