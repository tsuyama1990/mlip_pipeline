import logging
import sys
from pathlib import Path


def setup_logging(work_dir: Path | None = None, level: str = "INFO") -> None:
    """Configure centralized logging."""
    logger = logging.getLogger("mlip_autopipec")
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (if work_dir is provided)
    if work_dir:
        work_dir = Path(work_dir)
        # Ensure directory exists - though Orchestrator should handle this,
        # logger shouldn't crash if it doesn't.
        # But per memory, Orchestrator ensures it.
        # Here we just try to write if path is valid.
        if not work_dir.exists():
             # We might not want to create it here implicitly if Orchestrator does it.
             # But safe to ensure.
             work_dir.mkdir(parents=True, exist_ok=True)

        log_file = work_dir / "mlip_pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.info("Logging initialized.")
