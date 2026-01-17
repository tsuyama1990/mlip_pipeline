import logging
import sys
from pathlib import Path

# Try importing RichHandler, fall back to standard StreamHandler if not present
try:
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

def setup_logging(log_path: Path, level: str = "INFO") -> None:
    """
    Sets up the centralized logging system.
    Configures a file handler and a console handler (Rich if available).
    """
    # Create directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicates
    root_logger.handlers = []

    # File Handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    if HAS_RICH:
        console_handler = RichHandler(rich_tracebacks=True)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)

    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_path}")
