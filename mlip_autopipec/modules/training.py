"""Module for training a Pacemaker potential."""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write
from jinja2 import Template

from mlip_autopipec.config.models import TrainingConfig, TrainingRunMetrics
from mlip_autopipec.utils.ase_utils import read_training_data

log = logging.getLogger(__name__)


class TrainingFailedError(Exception):
    """Custom exception for errors during the training process."""


class NoTrainingDataError(Exception):
    """Custom exception for when no training data is available."""


class PacemakerTrainer:
    """Orchestrates the training of a Machine Learning Interatomic Potential."""

    def __init__(self, training_config: TrainingConfig) -> None:
        """Initialize the trainer with a validated configuration."""
        self.config = training_config

    def perform_training(self, generation: int) -> tuple[Path, TrainingRunMetrics]:
        """
        Executes the full training workflow:
        1. Reads data from the database.
        2. Prepares input files in a temporary directory.
        3. Runs the Pacemaker training executable.
        4. Parses metrics and moves the resulting potential.

        Args:
            generation: The current active learning generation index.

        Returns:
            A tuple containing the path to the trained potential file and the training metrics.

        Raises:
            NoTrainingDataError: If no data is found or database cannot be read.
            TrainingFailedError: If the training subprocess fails or output is invalid.
        """
        try:
            atoms_list = read_training_data(self.config.data_source_db)
        except ValueError as e:
            log.exception("Failed to read training data from database.")
            raise NoTrainingDataError(f"Error reading training data: {e}") from e

        if not atoms_list:
            msg = f"No training data found in '{self.config.data_source_db}'."
            log.error(msg)
            raise NoTrainingDataError(msg)

        # Use a temporary directory for the training process to avoid clutter
        # and ensure isolation. The context manager ensures cleanup.
        try:
            with tempfile.TemporaryDirectory(prefix="pacemaker_train_") as temp_dir_str:
                working_dir = Path(temp_dir_str)
                log.info(f"Preparing training input in {working_dir}...")
                self._prepare_pacemaker_input(atoms_list, working_dir)

                log.info("Executing Pacemaker training...")
                potential_path, rmse_forces, rmse_energy = self._execute_training(working_dir)

                final_path = Path.cwd() / potential_path.name
                shutil.move(potential_path, final_path)
                log.info(f"Potential saved to {final_path}")

                metrics = TrainingRunMetrics(
                    generation=generation,
                    num_structures=len(atoms_list),
                    rmse_forces=rmse_forces,
                    rmse_energy_per_atom=rmse_energy
                )
                return final_path, metrics
        except OSError as e:
            log.exception("Filesystem error during training.")
            raise TrainingFailedError(f"Filesystem error: {e}") from e

    def _prepare_pacemaker_input(self, training_data: list[Atoms], working_dir: Path) -> None:
        """Create the necessary input files for the Pacemaker executable."""
        data_file_path = working_dir / "training_data.xyz"

        # Ensure directory exists (TemporaryDirectory should do this, but just in case)
        working_dir.mkdir(parents=True, exist_ok=True)

        ase_write(data_file_path, training_data, format="extxyz")

        try:
            if self.config.template_file:
                with self.config.template_file.open() as f:
                    template_content = f.read()
            else:
                # Fallback or error if no template provided? Schema says Optional, but logic implies required.
                # Assuming config validation enforces it if needed, or providing a default string here.
                # For now, let's assume it might be None and raise if so.
                raise FileNotFoundError("No template file specified in configuration.")

            template = Template(template_content)
        except FileNotFoundError as e:
            log.exception(
                "Failed to open Jinja2 template file for Pacemaker.",
                extra={"template_file": self.config.template_file},
            )
            raise TrainingFailedError(
                f"Template file not found: {self.config.template_file}"
            ) from e

        rendered_config = template.render(config=self.config, data_file_path=str(data_file_path))

        config_file_path = working_dir / "pacemaker.in"
        config_file_path.write_text(rendered_config)

    def _execute_training(self, working_dir: Path) -> tuple[Path, float, float]:
        """Execute the `pacemaker` command in a secure subprocess."""
        executable = str(self.config.pacemaker_executable)
        if not shutil.which(executable):
            msg = f"Executable '{executable}' not found."
            log.error(msg)
            raise FileNotFoundError(msg)

        command = [executable]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                shell=False,
            )
        except subprocess.CalledProcessError as e:
            log.exception(
                "Pacemaker training subprocess failed.",
                extra={
                    "returncode": e.returncode,
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                },
            )
            msg = f"Pacemaker training failed with exit code {e.returncode}.\nStderr:\n{e.stderr}"
            raise TrainingFailedError(msg) from e

        match = re.search(r"Final potential saved to: (.*\.yace)", result.stdout)
        if not match:
            msg = "Could not find output potential file in training log."
            log.error(msg)
            raise TrainingFailedError(msg)

        potential_file = working_dir / match.group(1)
        if not potential_file.exists():
            msg = f"Potential file '{potential_file}' not found."
            log.error(msg)
            raise TrainingFailedError(msg)

        # Extract metrics
        rmse_forces = 0.0
        rmse_energy = 0.0

        match_f = re.search(r"RMSE forces:\s*([\d\.]+)", result.stdout)
        if match_f:
            rmse_forces = float(match_f.group(1))

        match_e = re.search(r"RMSE energy.*:\s*([\d\.]+)", result.stdout)
        if match_e:
            rmse_energy = float(match_e.group(1))

        return potential_file, rmse_forces, rmse_energy
