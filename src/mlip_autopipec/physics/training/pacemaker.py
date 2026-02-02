import logging
import os
import subprocess
from pathlib import Path

from mlip_autopipec.config import TrainingConfig
from mlip_autopipec.orchestration.interfaces import Trainer

logger = logging.getLogger(__name__)


class PacemakerTrainer(Trainer):
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(
        self, dataset: Path, previous_potential: Path | None, output_dir: Path
    ) -> Path:
        """
        Runs pace_train via subprocess or mock if environment variable PYACEMAKER_MOCK_MODE is set.
        """
        mock_mode = os.environ.get("PYACEMAKER_MOCK_MODE", "0") == "1"

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # In Spec example, output filename is "output_potential.yace" inside output_dir
        expected_output = output_dir / "output_potential.yace"

        if mock_mode:
            logger.info("Running PacemakerTrainer in MOCK MODE")
            expected_output.touch()
            return expected_output

        cmd = [self.config.command, "--dataset", str(dataset)]
        if previous_potential:
            cmd.extend(["--initial_potential", str(previous_potential)])

        cmd.extend(["--max_num_epochs", str(self.config.max_epochs)])
        cmd.extend(["--output_dir", str(output_dir)])

        logger.info(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")  # noqa: TRY400
            raise

        # Verify output exists
        if not expected_output.exists():
            msg = f"Expected output file {expected_output} was not created."
            logger.error(msg)
            raise FileNotFoundError(msg)

        return expected_output

    def update_dataset(self, new_data_paths: list[Path]) -> Path:
        """
        Updates the dataset with new data.
        For now, this is a placeholder that logs the update and returns the original dataset path.
        In a real implementation, this would merge new data into the dataset file.
        """
        logger.info(
            f"Updating dataset {self.config.dataset_path} with {len(new_data_paths)} new files."
        )
        # TODO: Implement actual dataset merging (requires ASE/Pandas/etc.)
        return self.config.dataset_path
