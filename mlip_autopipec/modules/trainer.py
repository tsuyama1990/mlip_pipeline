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

        This version adds support for writing a `force_mask` for per-atom
        force weighting if it is present in the `atoms.info` dictionary.

        Args:
            atoms_list: The list of structures to save.
            output_dir: The directory where the data file will be saved.

        Returns:
            The path to the created XYZ data file.

        """
        data_file = output_dir / "training_data.xyz"
        # Pacemaker expects the mask in the `info` dictionary of each Atoms object
        # when writing to extxyz.
        processed_atoms_list = []
        for atoms in atoms_list:
            new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
            if "mlip_force_mask" in new_atoms.info.get("key_value_pairs", {}):
                mask = new_atoms.info["key_value_pairs"]["mlip_force_mask"]
                # The key `pacemaker_force_mask` is specifically recognized by
                # the pacemaker-io fork of ASE.
                new_atoms.info["pacemaker_force_mask"] = mask
            processed_atoms_list.append(new_atoms)

        ase_write(data_file, processed_atoms_list, format="extxyz")
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
            # The command is executed as a list of arguments to prevent shell
            # injection, a critical security measure.
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
