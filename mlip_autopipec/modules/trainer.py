# ruff: noqa: D101, D102, D103, D107
"""Module for training a Pacemaker potential."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ase.io import write as ase_write

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.data.database import DatabaseManager
from mlip_autopipec.modules.config_generator import PacemakerConfigGenerator


class TrainingFailedError(Exception):
    """Custom exception for errors during the training process."""


class NoTrainingDataError(Exception):
    """Custom exception for when no training data is available."""


class PacemakerTrainer:
    """Orchestrates the training of a Pacemaker MLIP."""

    def __init__(self, config: SystemConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.config_generator = PacemakerConfigGenerator(config)
        self._temp_dir: Path | None = None

    def train(self) -> str:
        """Execute the full training workflow.

        Returns:
            The file path to the newly generated .yace potential file.

        """
        try:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="pacemaker_train_"))
            data_file_path = self._fetch_training_data(self._temp_dir)
            config_file_path = self.config_generator.generate_config(
                data_file_path, self._temp_dir
            )
            potential_path = self._execute_training(config_file_path, self._temp_dir)
            return potential_path
        finally:
            self._cleanup()

    def _fetch_training_data(self, output_dir: Path) -> Path:
        """Fetch training data from the database and save it to a temporary file.

        Args:
            output_dir: The directory where the data file will be saved.

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

        data_file = output_dir / "training_data.xyz"
        ase_write(data_file, atoms_list, format="extxyz")
        return data_file

    def _execute_training(self, config_file_path: Path, work_dir: Path) -> str:
        """Execute the pacemaker_train command as a subprocess.

        Args:
            config_file_path: The path to the Pacemaker config file.
            work_dir: The directory where the command will be executed.

        Raises:
            TrainingFailedError: If the subprocess returns a non-zero exit code.
            FileNotFoundError: If the 'pacemaker_train' executable is not found.

        Returns:
            The path to the generated .yace potential file.

        """
        executable = "pacemaker_train"
        if not shutil.which(executable):
            raise FileNotFoundError(
                f"Executable '{executable}' not found in PATH. "
                "Is pacemaker-ace installed correctly?"
            )

        command = [
            executable,
            "--config-file",
            str(config_file_path),
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=work_dir,
            )
        except subprocess.CalledProcessError as e:
            raise TrainingFailedError(f"Pacemaker training failed: {e.stderr}") from e

        # Parse stdout to find the path of the saved potential
        match = re.search(r"Final potential saved to: (.*\.yace)", result.stdout)
        if not match:
            raise TrainingFailedError(
                "Could not find the output potential file path in the training log."
            )

        # The path in the log is relative to the temp directory
        potential_file = work_dir / match.group(1)
        return str(potential_file)

    def _cleanup(self) -> None:
        """Remove the temporary directory and all its contents."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
