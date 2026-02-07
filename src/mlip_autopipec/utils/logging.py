import logging
from pathlib import Path


def setup_logging(workdir: Path, log_file: str = "mlip.log") -> None:
    log_path = workdir / log_file
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
