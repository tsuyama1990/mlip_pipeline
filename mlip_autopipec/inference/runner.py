"""
Runs inference (LAMMPS) simulations using the trained potential.
Detects uncertainty and aborts if necessary.
"""

import logging
import re
import shutil
import subprocess
from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.data_models.inference_models import InferenceResult
from mlip_autopipec.inference.writer import LammpsInputWriter

logger = logging.getLogger(__name__)


class LammpsRunner:
    """
    Orchestrates LAMMPS simulations.
    Handles input generation, execution, and output parsing (especially uncertainty).
    """

    def __init__(self, config: InferenceConfig, work_dir: Path) -> None:
        """
        Initialize the runner.

        Args:
            config: Inference configuration.
            work_dir: Directory to run simulations in.
        """
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.writer = LammpsInputWriter(config, work_dir)

    def run(self, atoms: Atoms, potential_path: Path) -> InferenceResult:
        """
        Run a LAMMPS simulation.

        Args:
            atoms: The starting structure.
            potential_path: Path to the .yace potential file.

        Returns:
            InferenceResult object containing success status, max uncertainty, and dump files.
        """
        try:
            # 1. Prepare Inputs
            input_file, data_file, log_file, dump_file = self.writer.write_inputs(atoms, potential_path)

            # 2. Check Executable
            executable = shutil.which(self.config.lammps_executable)
            if not executable:
                # If path is absolute/relative but not in PATH
                if Path(self.config.lammps_executable).exists():
                    executable = str(Path(self.config.lammps_executable).resolve())

            if not executable:
                msg = f"LAMMPS executable '{self.config.lammps_executable}' not found in PATH or is not executable."
                logger.error(msg)
                raise RuntimeError(msg)

            # 3. Build Command
            # "lmp -in input.lammps"
            cmd = [executable, "-in", str(input_file.name)]

            logger.info(f"Starting LAMMPS execution: {' '.join(cmd)}")

            # 4. Run
            # We use check=False to capture non-zero exit codes manually if needed
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=self.work_dir
            )

            if result.returncode != 0:
                logger.error(f"LAMMPS failed with exit code {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                return InferenceResult(
                    succeeded=False,
                    max_gamma_observed=0.0,
                    uncertain_structures=[]
                )

            # 5. Parse Output
            # We check the log file for thermodynamic output and max_gamma
            max_gamma = self._parse_max_gamma(log_file)
            logger.info(f"Max Gamma observed: {max_gamma}")

            uncertain_structures = []
            if max_gamma > self.config.uncertainty_threshold:
                logger.warning(f"Uncertainty threshold exceeded ({max_gamma} > {self.config.uncertainty_threshold})")
                if dump_file.exists() and dump_file.stat().st_size > 0:
                    uncertain_structures.append(dump_file)

            return InferenceResult(
                succeeded=True,
                max_gamma_observed=max_gamma,
                uncertain_structures=uncertain_structures
            )

        except Exception:
            logger.exception("An unexpected error occurred during LAMMPS execution")
            return InferenceResult(
                succeeded=False,
                max_gamma_observed=0.0,
                uncertain_structures=[]
            )

    def _parse_max_gamma(self, log_file: Path) -> float:
        """
        Parse the LAMMPS log file to find the maximum c_gamma value.
        Assumes "thermo_style custom ... c_gamma" was used.
        """
        if not log_file.exists():
            return 0.0

        max_val = 0.0
        try:
            with open(log_file) as f:
                content = f.read()

            # Look for lines with numbers.
            # This is heuristic. Better to parse column headers if possible.
            # But pacemaker/lammps output usually prints steps.
            # We can also rely on the dump file, but that's heavier.
            # Let's simple regex for now if we know the format or just parse all floats.

            # Alternative: if we ran with 'compute gamma', maybe we can just assume
            # if the run finished, we scan for "Max Gamma" if we printed it?
            # Our writer adds `thermo_style custom ... c_gamma`
            # So the last column might be gamma if configured last.

            # Simple fallback: return 0.0 if parsing fails, rely on dump file analysis if needed.
            # For this cycle, we assume the simulation might output it.

            # Let's iterate lines that look like thermo output
            lines = content.splitlines()
            # Find the header
            header_idx = -1
            gamma_col_idx = -1

            for i, line in enumerate(lines):
                if line.strip().startswith("Step"):
                    header_idx = i
                    parts = line.split()
                    if "c_gamma" in parts:
                        gamma_col_idx = parts.index("c_gamma")
                    break

            if header_idx != -1 and gamma_col_idx != -1:
                for line in lines[header_idx+1:]:
                    parts = line.split()
                    if len(parts) > gamma_col_idx:
                        try:
                            val = float(parts[gamma_col_idx])
                            if val > max_val:
                                max_val = val
                        except ValueError:
                            pass # loop section or other text

        except Exception:
            logger.warning("Failed to parse max_gamma from log.")

        return max_val
