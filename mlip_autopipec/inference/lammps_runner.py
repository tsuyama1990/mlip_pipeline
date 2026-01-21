"""
LAMMPS Runner Module.

This module provides the LammpsRunner class for executing Molecular Dynamics simulations.
It delegates input creation to LammpsInputWriter.
"""

import logging
import shutil
import subprocess
from pathlib import Path

from ase.atoms import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig, InferenceResult
from mlip_autopipec.inference.interfaces import MDRunner
from mlip_autopipec.inference.writer import LammpsInputWriter

logger = logging.getLogger(__name__)


class LammpsRunner(MDRunner):
    """
    Orchestrates LAMMPS simulations.
    Responsible for execution and result collection.
    """

    def __init__(self, config: InferenceConfig, work_dir: Path) -> None:
        """
        Initialize the runner.

        Args:
            config: Configuration object.
            work_dir: Directory to run simulations in.
        """
        self.config = config
        self.work_dir = work_dir
        self.writer = LammpsInputWriter(config, work_dir)

    def run(self, atoms: Atoms) -> InferenceResult:
        """
        Executes a LAMMPS simulation for the given atoms object.

        Args:
            atoms: The atomic structure to simulate.

        Returns:
            InferenceResult object containing simulation status and artifacts.
        """
        data_file = self.work_dir / "data.lammps"
        dump_file = self.work_dir / "dump.gamma"
        success = False

        try:
            # Delegate writing
            input_file, data_file_path, log_file, dump_file_path = self.writer.write_inputs(atoms)
            # Update paths in case writer changed them
            data_file = data_file_path
            dump_file = dump_file_path

            # Determine executable
            executable = (
                str(self.config.lammps_executable)
                if self.config.lammps_executable
                else "lmp_serial"
            )

            # Verify executable exists and is executable
            if not shutil.which(executable):
                logger.error(
                    f"LAMMPS executable '{executable}' not found in PATH or is not executable."
                )
                return InferenceResult(
                    succeeded=False,
                    final_structure=None,
                    uncertain_structures=[],
                    max_gamma_observed=0.0,
                )

            # Execute
            cmd = [
                executable,
                "-in",
                str(input_file),
                "-log",
                str(log_file),
            ]

            logger.info(f"Starting LAMMPS execution: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, cwd=self.work_dir
            )

            logger.info("LAMMPS execution completed successfully.")
            success = True

        except subprocess.CalledProcessError as e:
            logger.error(f"LAMMPS execution failed with return code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            success = False
        except Exception:
            logger.exception("An unexpected error occurred during LAMMPS execution.")
            success = False

        # Process Results (Best Effort)
        uncertain_structures = []
        if success and dump_file.exists() and dump_file.stat().st_size > 0:
            uncertain_structures.append(dump_file)

        max_gamma = 0.0

        return InferenceResult(
            succeeded=success,
            final_structure=data_file if success else None,
            uncertain_structures=uncertain_structures,
            max_gamma_observed=max_gamma,
        )
