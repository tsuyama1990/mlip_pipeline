# ruff: noqa: D101, D102, D103, D107
"""Module for training a Pacemaker potential."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
from ase.io import write as ase_write

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.data.database import DatabaseManager


class TrainingFailedError(Exception):
    """Custom exception for errors during the training process."""


class NoTrainingDataError(Exception):
    """Custom exception for when no training data is available."""


class PacemakerTrainer:
    """Orchestrates the training of a Pacemaker MLIP."""

    def __init__(self, config: SystemConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self._temp_dir: Path | None = None

    def train(self) -> str:
        """Execute the full training workflow.

        Returns:
            The file path to the newly generated .yace potential file.

        """
        try:
            data_file_path = self._fetch_training_data()
            config_file_path = self._generate_pacemaker_config(data_file_path)
            potential_path = self._execute_training(config_file_path)
            return potential_path
        finally:
            self._cleanup()

    def _fetch_training_data(self) -> Path:
        """Fetch training data from the database and save it to a temporary file.

        Raises:
            NoTrainingDataError: If no completed calculations are found in the database.

        Returns:
            The path to the temporary XYZ data file.

        """
        atoms_list = self.db_manager.get_completed_calculations()
        if not atoms_list:
            raise NoTrainingDataError(
                "No completed DFT calculations found in the database."
            )

        self._temp_dir = Path(tempfile.mkdtemp(prefix="pacemaker_train_"))
        data_file = self._temp_dir / "training_data.xyz"
        ase_write(data_file, atoms_list, format="extxyz")
        return data_file

    def _generate_pacemaker_config(self, data_file_path: Path) -> Path:
        """Generate the Pacemaker YAML config file from the SystemConfig.

        Args:
            data_file_path: The path to the training data file.

        Returns:
            The path to the generated YAML configuration file.

        """
        if self._temp_dir is None:
            # This should not happen in the normal workflow
            raise RuntimeError("Temporary directory not created.")

        trainer_params = self.config.trainer
        config_dict = {
            "fit_params": {
                "dataset_filename": str(data_file_path),
                "loss_weights": {
                    "energy": trainer_params.loss_weights.energy,
                    "forces": trainer_params.loss_weights.forces,
                    "stress": trainer_params.loss_weights.stress,
                },
                "ace": {
                    "radial_basis": trainer_params.ace_params.radial_basis,
                    "correlation_order": trainer_params.ace_params.correlation_order,
                    "element_dependent_cutoffs": (
                        trainer_params.ace_params.element_dependent_cutoffs
                    ),
                },
            }
        }

        config_file = Path(self._temp_dir) / "pacemaker_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return config_file

    def _execute_training(self, config_file_path: Path) -> str:
        """Execute the pacemaker_train command as a subprocess.

        Args:
            config_file_path: The path to the Pacemaker config file.

        Raises:
            TrainingFailedError: If the subprocess returns a non-zero exit code.

        Returns:
            The path to the generated .yace potential file.

        """
        if self._temp_dir is None:
            # This should not happen in the normal workflow
            raise RuntimeError("Temporary directory not created before training.")

        command = [
            "pacemaker_train",
            "--config-file",
            str(config_file_path),
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=self._temp_dir,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            stderr = e.stderr if hasattr(e, "stderr") else str(e)
            raise TrainingFailedError(f"Pacemaker training failed: {stderr}") from e

        # Parse stdout to find the path of the saved potential
        match = re.search(r"Final potential saved to: (.*\.yace)", result.stdout)
        if not match:
            raise TrainingFailedError(
                "Could not find the output potential file path in the training log."
            )

        # The path in the log is relative to the temp directory
        potential_file = self._temp_dir / match.group(1)
        return str(potential_file)

    def _cleanup(self) -> None:
        """Remove the temporary directory and all its contents."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
