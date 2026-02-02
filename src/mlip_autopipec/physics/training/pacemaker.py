import logging
import os
import subprocess
from pathlib import Path

from mlip_autopipec.config import TrainingConfig

logger = logging.getLogger(__name__)


class PacemakerTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, dataset: Path, previous_potential: Path | None = None) -> Path:
        """
        Runs pace_train via subprocess or mock if environment variable PYACEMAKER_MOCK_MODE is set.
        """
        mock_mode = os.environ.get("PYACEMAKER_MOCK_MODE", "0") == "1"

        output_file = Path("output.yace")

        if mock_mode:
            logger.info("Running PacemakerTrainer in MOCK MODE")
            output_file.touch()
            return output_file

        cmd = [self.config.command, "--dataset", str(dataset)]
        if previous_potential:
            cmd.extend(["--initial-potential", str(previous_potential)])

        cmd.extend(["--max-epochs", str(self.config.max_epochs)])

        logger.info(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")  # noqa: TRY400
            raise

        # Verify output exists
        if not output_file.exists():
            # In a real scenario, we might need to check what the output filename is
            # For now, we assume standard behavior or raise error
            msg = f"Expected output file {output_file} was not created."
            logger.error(msg)
            raise FileNotFoundError(msg)

        return output_file
