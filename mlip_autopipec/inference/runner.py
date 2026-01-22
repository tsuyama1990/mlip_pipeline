"""
LAMMPS Runner Module.

This module provides the LammpsRunner class for executing Molecular Dynamics simulations.
It delegates input creation to LammpsInputWriter.
"""

import logging
import re
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

    def run(self, atoms: Atoms, potential_path: Path) -> InferenceResult:
        """
        Executes a LAMMPS simulation for the given atoms object.

        Args:
            atoms: The atomic structure to simulate.
            potential_path: Path to the .yace potential.

        Returns:
            InferenceResult object containing simulation status and artifacts.
        """
        success = False
        max_gamma = 0.0

        # Initialize paths (will be overwritten by writer)
        data_file = self.work_dir / "data.lammps"
        dump_file = self.work_dir / "dump.gamma"
        log_file = self.work_dir / "log.lammps"

        try:
            # Delegate writing
            input_file, data_file_path, log_file_path, dump_file_path = self.writer.write_inputs(atoms, potential_path)
            data_file = data_file_path
            dump_file = dump_file_path
            log_file = log_file_path

            # Determine executable
            executable = (
                str(self.config.lammps_executable)
                if self.config.lammps_executable
                else "lmp_serial"
            )

            # Verify executable exists and is executable
            if not shutil.which(executable):
                msg = f"LAMMPS executable '{executable}' not found in PATH or is not executable."
                logger.error(msg)
                raise RuntimeError(msg)

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

        except Exception as e:
            logger.exception(f"An unexpected error occurred during LAMMPS execution: {e}")
            success = False

        # Parse Log for Max Gamma
        if log_file.exists():
            max_gamma = self._parse_max_gamma(log_file)
            logger.info(f"Max Gamma observed: {max_gamma}")

        # Process Results
        uncertain_structures = []
        if dump_file.exists() and dump_file.stat().st_size > 0:
            # If we have a dump, it means we ran.
            # We treat the dump as the source of uncertain structures if gamma > threshold.
            uncertain_structures.append(dump_file)

        return InferenceResult(
            succeeded=success,
            final_structure=data_file if success else None,
            uncertain_structures=uncertain_structures,
            max_gamma_observed=max_gamma,
        )

    def _parse_max_gamma(self, log_file: Path) -> float:
        """Parses the log file to find the maximum gamma value recorded."""
        max_g = 0.0
        try:
            content = log_file.read_text()
            lines = content.splitlines()
            header_found = False
            for line in lines:
                if "Step" in line and "c_max_gamma" in line:
                    header_found = True
                    continue
                if header_found:
                    if "Loop time" in line:
                        break
                    parts = line.split()
                    if len(parts) >= 5: # We have at least 5 cols
                        try:
                            # last col is c_max_gamma
                            val = float(parts[-1])
                            if val > max_g:
                                max_g = val
                        except ValueError:
                            pass
        except Exception:
            logger.warning("Failed to parse max gamma from log.")

        return max_g
