import logging
import sys

from mlip_autopipec.domain_models import GlobalConfig


def setup_logging(config: GlobalConfig) -> logging.Logger:
    logger = logging.getLogger("mlip_autopipec")
    logger.setLevel(logging.DEBUG)  # Capture all, handlers filter

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    work_dir = config.orchestrator.work_dir
    # Ensure directory exists for logging
    if not work_dir.exists():
         work_dir.mkdir(parents=True, exist_ok=True)

    log_file = work_dir / "mlip_pipeline.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
