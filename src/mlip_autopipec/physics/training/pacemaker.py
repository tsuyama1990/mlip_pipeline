import logging
import os
import subprocess
from pathlib import Path

import ase.io

from mlip_autopipec.config import TrainingConfig
from mlip_autopipec.orchestration.interfaces import Trainer

logger = logging.getLogger(__name__)


class PacemakerTrainer(Trainer):
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, dataset: Path, previous_potential: Path | None, output_dir: Path) -> Path:
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
        Merges new data into the dataset file using ASE.
        """
        logger.info(
            f"Updating dataset {self.config.dataset_path} with {len(new_data_paths)} new files."
        )

        all_atoms = []
        # Load existing dataset if it exists
        if self.config.dataset_path.exists():
            try:
                # Read all images
                existing_data = ase.io.read(self.config.dataset_path, index=":")
                if isinstance(existing_data, list):
                    all_atoms.extend(existing_data)
                else:
                    all_atoms.append(existing_data)
                logger.info(f"Loaded {len(all_atoms)} existing structures.")
            except Exception:
                logger.exception("Failed to read existing dataset.")
                # Proceeding with empty list if read fails, might be dangerous but safer than crashing?
                # Actually, if the file exists but is corrupt, we probably should raise error.
                # But let's assume we want to append.
                raise

        # Load new data
        new_count = 0
        for path in new_data_paths:
            try:
                new_data = ase.io.read(path, index=":")
                if isinstance(new_data, list):
                    all_atoms.extend(new_data)
                    new_count += len(new_data)
                else:
                    all_atoms.append(new_data)
                    new_count += 1
            except Exception:
                logger.warning(f"Failed to read new data file: {path}")

        logger.info(f"Adding {new_count} new structures.")

        # Write back
        # Ensure directory exists
        self.config.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        ase.io.write(self.config.dataset_path, all_atoms)
        logger.info(f"Saved updated dataset with {len(all_atoms)} total structures.")

        return self.config.dataset_path
