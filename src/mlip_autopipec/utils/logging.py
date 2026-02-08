import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level.upper() not in valid_levels:
        msg = f"Invalid log level: {level}. Must be one of {valid_levels}"
        raise ValueError(msg)

    numeric_level = getattr(logging, level.upper(), None)

    # Clear existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
