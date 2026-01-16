"""Module for training a Pacemaker potential."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.db import connect as ase_db_connect
from ase.io import write as ase_write
from jinja2 import Template

from mlip_autopipec.config.training import TrainingConfig


class TrainingFailedError(Exception):
    """Custom exception for errors during the training process."""


class NoTrainingDataError(Exception):
    """Custom exception for when no training data is available."""


class PacemakerTrainer:
    """Orchestrates the training of a Machine Learning Interatomic Potential."""

    def __init__(self, training_config: TrainingConfig) -> None:
        """Initialize the trainer with a validated configuration."""
        self.config = training_config

    def train(self) -> Path:
        """Execute the full training workflow."""
        atoms_list = self._read_data_from_db()
        if not atoms_list:
            msg = f"No training data found in '{self.config.data_source_db}'."
            raise NoTrainingDataError(msg)

        with tempfile.TemporaryDirectory(prefix="pacemaker_train_") as temp_dir_str:
            working_dir = Path(temp_dir_str)
            self._prepare_pacemaker_input(atoms_list, working_dir)
            potential_path = self._execute_training(working_dir)
            final_path = Path.cwd() / potential_path.name
            shutil.move(potential_path, final_path)
            return final_path

    def _read_data_from_db(self) -> list[Atoms]:
        """Read all atomic structures from the configured ASE database."""
        atoms_list = []
        with ase_db_connect(self.config.data_source_db) as db:
            for row in db.select():
                atoms = row.toatoms()
                if hasattr(row, "data") and row.data:
                    if "energy" in row.data:
                        atoms.info["energy"] = row.data["energy"]
                    if "forces" in row.data:
                        atoms.arrays["forces"] = np.array(row.data["forces"])
                atoms_list.append(atoms)
        return atoms_list

    def _prepare_pacemaker_input(
        self, training_data: list[Atoms], working_dir: Path
    ) -> None:
        """Create the necessary input files for the Pacemaker executable."""
        data_file_path = working_dir / "training_data.xyz"
        ase_write(data_file_path, training_data, format="extxyz")

        with self.config.template_file.open() as f:
            template = Template(f.read())

        rendered_config = template.render(
            config=self.config, data_file_path=str(data_file_path)
        )

        config_file_path = working_dir / "pacemaker.in"
        config_file_path.write_text(rendered_config)

    def _execute_training(self, working_dir: Path) -> Path:
        """Execute the `pacemaker` command in a secure subprocess."""
        executable = str(self.config.pacemaker_executable)
        if not shutil.which(executable):
            msg = f"Executable '{executable}' not found."
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
            msg = f"Pacemaker training failed with exit code {e.returncode}.\nStderr:\n{e.stderr}"
            raise TrainingFailedError(msg) from e

        match = re.search(r"Final potential saved to: (.*\.yace)", result.stdout)
        if not match:
            msg = "Could not find output potential file in training log."
            raise TrainingFailedError(msg)

        potential_file = working_dir / match.group(1)
        if not potential_file.exists():
            msg = f"Potential file '{potential_file}' not found."
            raise TrainingFailedError(msg)

        return potential_file
