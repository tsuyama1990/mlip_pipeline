import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure global logging settings.
    Uses force=True to ensure reliable log capture during CLI testing.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        msg = f"Invalid log level: {level}"
        raise ValueError(msg) # noqa: TRY004

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )
