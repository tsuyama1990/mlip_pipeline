import contextlib
import logging
import shlex
import subprocess
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.domain_models.inference_models import InferenceResult
from mlip_autopipec.inference.parsers import LammpsLogParser

logger = logging.getLogger(__name__)


class LammpsRunner:
    """
    Handles execution of LAMMPS MD simulations.
    Ensures input validation and security.
    """

    def __init__(self, config: InferenceConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, atoms: Atoms, potential_path: Path, uid: str) -> InferenceResult:
        """
        Runs LAMMPS simulation for a single structure.
        """
        # Validate inputs
        if not isinstance(atoms, Atoms):
            raise TypeError(f"Expected ase.Atoms object, got {type(atoms)}")

        if not potential_path.exists():
            msg = f"Potential file not found: {potential_path}"
            raise FileNotFoundError(msg)

        try:
            # 1. Prepare Inputs
            self._write_inputs(atoms, potential_path, uid)

            # 2. Execute LAMMPS
            log_file = self.work_dir / f"{uid}.log"
            stdout_file = self.work_dir / f"{uid}.stdout"
            stderr_file = self.work_dir / f"{uid}.stderr"

            cmd = self._build_command(uid)

            # Security: Ensure shell=False
            with open(stdout_file, "w") as f_out, open(stderr_file, "w") as f_err:
                process = subprocess.run(
                    cmd,
                    cwd=self.work_dir,
                    stdout=f_out,
                    stderr=f_err,
                    check=False,
                    shell=False
                )

            # 3. Parse Results
            if process.returncode != 0:
                stderr_content = ""
                if stderr_file.exists():
                     with contextlib.suppress(OSError):
                         with open(stderr_file, "rb") as f:
                             f.seek(-1024, 2)
                             stderr_content = f.read().decode("utf-8", errors="replace")

                logger.warning(f"LAMMPS failed for {uid} with code {process.returncode}. Stderr: {stderr_content}")
                return InferenceResult(
                    uid=uid,
                    succeeded=False,
                    error_message=f"Process exited with code {process.returncode}"
                )

            # Parse log file for gamma/uncertainty
            max_gamma, halted, step = LammpsLogParser.parse(log_file)

            return InferenceResult(
                uid=uid,
                succeeded=True,
                max_gamma_observed=max_gamma,
                halted=halted,
                halt_step=step
            )

        except Exception as e:
            logger.exception(f"Unexpected error in LammpsRunner for {uid}")
            return InferenceResult(uid=uid, succeeded=False, error_message=str(e))

    def _write_inputs(self, atoms: Atoms, potential_path: Path, uid: str) -> None:
        """Writes data file and input script."""
        data_file = self.work_dir / f"{uid}.data"
        # type: ignore[no-untyped-call]
        atoms.write(str(data_file), format="lammps-data")

        input_script = self.work_dir / f"{uid}.in"
        content = self._generate_input_script(data_file.name, potential_path, uid)
        input_script.write_text(content)

    def _generate_input_script(self, data_file: str, potential_path: Path, uid: str) -> str:
        # Simplified template
        return f"""
        units metal
        atom_style atomic
        boundary p p p
        read_data {data_file}
        pair_style pace
        pair_coeff * * {potential_path.absolute()} Al

        thermo 10
        thermo_style custom step temp pe ke etotal press

        # MD settings from config
        timestep {self.config.timestep}
        fix 1 all nvt temp {self.config.temperature} {self.config.temperature} 0.1

        run {self.config.steps}
        """

    def _build_command(self, uid: str) -> list[str]:
        """Builds the execution command."""
        exe = self.config.lammps_executable
        if not exe:
            raise ValueError("LAMMPS executable not configured")

        exe_str = str(exe)

        # Security: basic validation
        if ";" in exe_str or "|" in exe_str:
             raise ValueError("Unsafe characters in LAMMPS command")

        parts = shlex.split(exe_str)
        parts.extend(["-in", f"{uid}.in", "-log", f"{uid}.log"])
        return parts
