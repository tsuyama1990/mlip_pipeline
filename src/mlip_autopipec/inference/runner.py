"""
Runs inference (LAMMPS) simulations using the trained potential.
Detects uncertainty and aborts if necessary.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from ase import Atoms

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

            # 2. Validate and Resolve Executable
            executable = self._resolve_executable()

            # 3. Build Command (Whitelist approach: only allow fixed flag structure)
            # cmd structure: [executable, "-in", input_file_name]
            # No user-provided shell strings are passed directly.
            cmd = [str(executable), "-in", str(input_file.name)]

            logger.info(f"Starting LAMMPS execution: {' '.join(cmd)}")

            # 4. Run
            # shell=False is critical for security to prevent shell injection.
            # We strictly control the arguments list.
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=self.work_dir,
                shell=False
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

    def _resolve_executable(self) -> str:
        """
        Resolves the LAMMPS executable path and performs security checks.
        """
        raw_path = str(self.config.lammps_executable)

        # Security: Basic character blacklist for paranoid validation,
        # though subprocess.run(shell=False) handles this safely.
        # We reject obviously suspicious characters in the path itself.
        if any(char in raw_path for char in [";", "|", "&", "`", "$", "(", ")"]):
             raise ValueError(f"Security: Invalid characters detected in executable path: {raw_path}")

        executable = shutil.which(raw_path)
        if not executable:
            # If path is absolute/relative but not in PATH
            p = Path(raw_path)
            if p.exists() and p.is_file(): # Ensure it is a file
                 # Ensure executable permission
                 if not os.access(p, os.X_OK):
                     raise ValueError(f"File at {p} is not executable.")
                 executable = str(p.resolve())
            else:
                 msg = f"LAMMPS executable '{raw_path}' not found in PATH or is not a valid file."
                 logger.error(msg)
                 raise RuntimeError(msg)

        return executable

    def _parse_max_gamma(self, log_file: Path) -> float:
        """
        Parse the LAMMPS log file to find the maximum c_gamma value.
        """
        if not log_file.exists():
            return 0.0

        max_val = 0.0
        try:
            with log_file.open() as f:
                content = f.read()

            lines = content.splitlines()
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
                            max_val = max(max_val, val)
                        except ValueError:
                            pass

        except Exception:
            logger.warning("Failed to parse max_gamma from log.")

        return max_val
