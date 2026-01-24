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
from mlip_autopipec.inference.parsers import LogParser
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
        # Validate inputs
        if not isinstance(atoms, Atoms):
            raise TypeError(f"Expected ase.Atoms object, got {type(atoms)}")

        if not potential_path.exists():
            raise FileNotFoundError(f"Potential file not found: {potential_path}")

        try:
            # 1. Prepare Inputs
            input_file, data_file, log_file, dump_file = self.writer.write_inputs(atoms, potential_path)
            stdout_file = self.work_dir / "stdout.log"
            stderr_file = self.work_dir / "stderr.log"

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
            # We redirect stdout/stderr to files to prevent OOM on large outputs (Scalability).
            with stdout_file.open("w") as f_out, stderr_file.open("w") as f_err:
                result = subprocess.run(  # noqa: S603
                    cmd,
                    check=False,
                    stdout=f_out,
                    stderr=f_err,
                    cwd=self.work_dir,
                    shell=False
                )

            # 5. Parse Output
            max_gamma, halted, halt_step = LogParser.parse(log_file)
            logger.info(f"Max Gamma observed: {max_gamma}")

            if halted:
                logger.warning(f"Simulation halted at step {halt_step} due to high uncertainty.")

            # Determine success
            # If returncode is 0, it succeeded.
            # If returncode != 0, it ONLY succeeds if halted=True (watchdog trigger).
            if result.returncode != 0 and not halted:
                logger.error(f"LAMMPS failed with exit code {result.returncode}")
                # Log last few lines of stderr if available
                if stderr_file.exists():
                     try:
                         # Read only last 1KB
                         with stderr_file.open("rb") as f:
                             try:
                                 f.seek(-1024, os.SEEK_END)
                             except OSError:
                                 pass # File smaller than 1KB
                             tail = f.read().decode("utf-8", errors="replace")
                             logger.error(f"Stderr tail: {tail}")
                     except Exception:
                         pass

                return InferenceResult(
                    succeeded=False,
                    max_gamma_observed=max_gamma,
                    uncertain_structures=[]
                )

            uncertain_structures = []
            if (
                max_gamma > self.config.uncertainty_threshold
                and dump_file.exists()
                and dump_file.stat().st_size > 0
            ):
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
        If config.lammps_executable is None, attempts to find 'lmp' or 'lmp_serial' in PATH.
        """
        raw_path = str(self.config.lammps_executable) if self.config.lammps_executable else None

        if not raw_path:
            # Fallback to standard names
            for name in ["lmp", "lmp_serial", "lmp_mpi"]:
                found = shutil.which(name)
                if found:
                    raw_path = found
                    break

            if not raw_path:
                 msg = "LAMMPS executable not specified and not found in PATH."
                 raise RuntimeError(msg)

        # Check blacklist characters if path was provided
        if any(char in raw_path for char in [";", "|", "&", "`", "$", "(", ")"]):
             msg = f"Security: Invalid characters detected in executable path: {raw_path}"
             raise ValueError(msg)

        executable = shutil.which(raw_path)
        if not executable:
            # If path is absolute/relative but not in PATH
            p = Path(raw_path)
            if p.exists() and p.is_file(): # Ensure it is a file
                 # Ensure executable permission
                 if not os.access(p, os.X_OK):
                     msg = f"File at {p} is not executable."
                     raise ValueError(msg)
                 executable = str(p.resolve())
            else:
                 msg = f"LAMMPS executable '{raw_path}' not found in PATH or is not a valid file."
                 logger.error(msg)
                 raise RuntimeError(msg)

        return executable
