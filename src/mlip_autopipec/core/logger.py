import logging
import sys
from pathlib import Path

from mlip_autopipec.constants import LOG_DATE_FORMAT, LOG_FORMAT


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(work_dir: Path | None = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler
    if work_dir:
        log_file = work_dir / "mlip_pipeline.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)
