# ruff: noqa: D101, D102, D103, D107
"""Module for training a Pacemaker potential."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.config_generator import PacemakerConfigGenerator


class TrainingFailedError(Exception):
    """Custom exception for errors during the training process."""


class NoTrainingDataError(Exception):
    """Custom exception for when no training data is available."""


class PacemakerTrainer:
    """Orchestrates the training of a Machine Learning Interatomic Potential (MLIP).

    This class is responsible for managing the end-to-end training workflow. It
    takes a list of atomic structures, prepares them for training, generates the
    necessary configuration for the Pacemaker engine, and executes the training
    process in a secure subprocess.

    Attributes:
        config: The system configuration object.
        config_generator: A generator object for creating Pacemaker config files.

    """

    def __init__(
        self, config: SystemConfig, config_generator: PacemakerConfigGenerator
    ):
        self.config = config
        self.config_generator = config_generator
        self._temp_dir: Path | None = None

    def train(self, atoms_list: list[Atoms]) -> str:
        """Execute the full training workflow for a given set of atomic structures.

        This method orchestrates the entire training process: it creates a temporary
        directory for intermediate files, prepares the training data, generates the
        Pacemaker configuration, and executes the training command.

        Args:
            atoms_list: A list of ASE Atoms objects to be used as training data.

        Returns:
            The absolute file path to the newly generated .yace potential file.

        Raises:
            NoTrainingDataError: If the provided `atoms_list` is empty.

        """
        if not atoms_list:
            raise NoTrainingDataError("The provided list of structures is empty.")

        try:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="pacemaker_train_"))
            data_file_path = self._prepare_training_data(atoms_list, self._temp_dir)
            config_file_path = self.config_generator.generate_config(
                data_file_path, self._temp_dir
            )
            potential_path = self._execute_training(config_file_path, self._temp_dir)
            return potential_path
        finally:
            self._cleanup()

    def _prepare_training_data(self, atoms_list: list[Atoms], output_dir: Path) -> Path:
        """Save the provided list of Atoms to a temporary file in extxyz format.

        This method now also handles the `force_mask` if it is present in the
        `atoms.info` dictionary, writing it as a per-atom property.

        Args:
            atoms_list: The list of structures to save.
            output_dir: The directory where the data file will be saved.

        Returns:
            The path to the created XYZ data file.

        """
        import numpy as np

        data_file = output_dir / "training_data.xyz"

        # Prepare a list of atoms to write, adding the force_mask as an array
        atoms_to_write = []
        for atoms in atoms_list:
            writable_atoms = atoms.copy()
            if "force_mask" in writable_atoms.info:
                mask = writable_atoms.info["force_mask"]
                # Ensure the mask is a 2D array for ASE to write correctly
                if mask.ndim == 1:
                    mask = mask[:, np.newaxis]
                writable_atoms.new_array("force_mask", mask)
                del writable_atoms.info["force_mask"]  # Clean up info dict
            atoms_to_write.append(writable_atoms)

        ase_write(data_file, atoms_to_write, format="extxyz")
        return data_file

    def _execute_training(self, config_file_path: Path, work_dir: Path) -> str:
        """Execute the `pacemaker_train` command in a secure subprocess.

        This method first verifies that the `pacemaker_train` executable is available
        in the system's PATH. It then constructs the command as a list of arguments
        to prevent shell injection vulnerabilities. The subprocess is executed in a
        temporary working directory to isolate its output.

        Args:
            config_file_path: The path to the Pacemaker config file.
            work_dir: The directory where the command will be executed.

        Raises:
            TrainingFailedError: If the subprocess returns a non-zero exit code or if
                the output log cannot be parsed.
            FileNotFoundError: If the 'pacemaker_train' executable is not found.

        Returns:
            The absolute path to the generated .yace potential file.

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
            # The command is executed as a list of arguments, which is a security
            # best practice that prevents shell injection vulnerabilities. The stderr
            # is included in the exception message to provide detailed, actionable
            # feedback to the user, which is crucial for debugging scientific workflows.
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
