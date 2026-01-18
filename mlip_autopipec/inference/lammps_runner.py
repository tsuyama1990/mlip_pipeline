"""
LAMMPS Runner Module.

This module provides the LammpsRunner class for executing Molecular Dynamics simulations.
It delegates input creation to LammpsInputWriter.
"""

import logging
import subprocess
from pathlib import Path

from ase.atoms import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig, InferenceResult
from mlip_autopipec.inference.writer import LammpsInputWriter

logger = logging.getLogger(__name__)


class LammpsRunner:
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
        try:
            # Delegate writing
            input_file, data_file, log_file, dump_file = self.writer.write_inputs(atoms)

            # Execute
            cmd = [
                str(self.config.lammps_executable)
                if self.config.lammps_executable
                else "lmp_serial",
                "-in",
                str(input_file),
                "-log",
                str(log_file),
            ]

            logger.info(f"Starting LAMMPS execution: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, cwd=self.work_dir
            )

            if result.returncode != 0:
                logger.error(f"LAMMPS execution failed with return code {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                success = False
            else:
                logger.info("LAMMPS execution completed successfully.")
                success = True

        except (subprocess.SubprocessError, OSError):
            logger.exception("An error occurred during LAMMPS execution setup or run.")
            success = False
            # Ensure variables are defined if exception occurs before assignment
            # Though writer.write_inputs usually succeeds or raises.
            # If writer fails, we might not have file paths.
            # We can't return paths if they weren't defined.
            # Initialize defaults
            data_file = self.work_dir / "data.lammps"
            dump_file = self.work_dir / "dump.gamma"

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
