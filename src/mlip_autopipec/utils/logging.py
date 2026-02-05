import logging
import sys


def setup_logging(level: str = "INFO", force: bool = True) -> None:
    """
    Sets up the logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        force: If True, forces reconfiguration of logging (useful for testing).
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        msg = f"Invalid log level: {level}"
        raise TypeError(msg)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=force
    )

    # Set levels for some noisy libraries if needed
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
