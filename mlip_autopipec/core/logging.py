"""
Centralized logging configuration for the MLIP-AutoPipe project.
"""
import logging
import sys
from pathlib import Path
from mlip_autopipec.exceptions import LoggingError

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

    Args:
        log_path: Path where the log file will be written.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Raises:
        LoggingError: If log directory cannot be created or file handler fails.
    """
    try:
        # Create directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise LoggingError(f"Failed to create log directory {log_path.parent}: {e}") from e

    root_logger = logging.getLogger()
    try:
        root_logger.setLevel(getattr(logging, level.upper()))
    except AttributeError:
        root_logger.setLevel(logging.INFO)
        logging.warning(f"Invalid logging level '{level}'. Defaulting to INFO.")

    # Clear existing handlers to avoid duplicates
    root_logger.handlers = []

    # File Handler
    try:
        file_handler = logging.FileHandler(log_path, mode='a')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        raise LoggingError(f"Failed to setup file logging at {log_path}: {e}") from e

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
