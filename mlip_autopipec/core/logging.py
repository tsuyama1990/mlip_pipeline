import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_path: Path, level: str | None = "INFO") -> None:
    """
    Sets up the centralized logging configuration.

    Args:
        log_path: Path to the log file.
        level: Logging level (default: "INFO").
    """
    # Ensure log directory exists
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root logger
    # We want to capture logs from our package and maybe others.
    # Spec says "Ensure that all modules use the same logging instance."
    # Usually this means configuring the root logger or a package logger.

    # Let's configure the package logger to avoid interfering with other libs too much,
    # but SPEC says "Sets up the Python logger... ensures all modules use the same logging instance."

    # We will configure the root logger but restricted to our app context or general enough.
    # Actually, for a CLI app, configuring root logger is fine.

    logging_level = getattr(logging, level.upper()) if level else logging.INFO

    # Handlers
    handlers: list[logging.Handler] = []

    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging_level)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # Console Handler (Rich)
    # Using RichHandler for beautiful output
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setLevel(logging_level)
    handlers.append(rich_handler)

    # Configuration
    logging.basicConfig(
        level=logging_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,  # Reconfigure if already configured
    )

    # Silence noisy libraries if needed (optional)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
