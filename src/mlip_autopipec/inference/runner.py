import contextlib
import logging
import subprocess
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.data_models.inference_models import InferenceResult
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
                    check=False,  # We handle return code manually
                    shell=False,
                )

            # 3. Parse Results
            if process.returncode != 0:
                # Log stderr content for debugging
                stderr_content = ""
                if stderr_file.exists():
                    # Read only last 1KB safely
                    with contextlib.suppress(OSError):
                        with open(stderr_file, "rb") as f:
                            f.seek(-1024, 2)  # Seek from end
                            stderr_content = f.read().decode("utf-8", errors="replace")

                logger.warning(
                    f"LAMMPS failed for {uid} with code {process.returncode}. Stderr: {stderr_content}"
                )
                return InferenceResult(
                    uid=uid,
                    success=False,
                    error_message=f"Process exited with code {process.returncode}",
                )

            # Parse log file for gamma/uncertainty
            max_gamma, halted, step = LammpsLogParser.parse(log_file)

            return InferenceResult(
                uid=uid, success=True, max_gamma=max_gamma, halted=halted, halt_step=step
            )

        except Exception as e:
            logger.exception(f"Unexpected error in LammpsRunner for {uid}")
            return InferenceResult(uid=uid, success=False, error_message=str(e))

    def _write_inputs(self, atoms: Atoms, potential_path: Path, uid: str) -> None:
        """Writes data file and input script."""
        # Write data file
        data_file = self.work_dir / f"{uid}.data"
        # Type check ignore because external lib
        # type: ignore[no-untyped-call]
        atoms.write(str(data_file), format="lammps-data")

        # Write input script
        input_script = self.work_dir / f"{uid}.in"
        content = self._generate_input_script(data_file.name, potential_path, uid)
        input_script.write_text(content)

    def _generate_input_script(self, data_file: str, potential_path: Path, uid: str) -> str:
        """Generates LAMMPS input script content."""
        # This is a simplified template. In real impl, use jinja2 or config params.
        # Use abs path for potential if needed, or symlink
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

        run {self.config.n_steps}
        """

    def _build_command(self, uid: str) -> list[str]:
        """Builds the execution command."""
        exe = self.config.lammps_command
        if not exe:
            raise ValueError("LAMMPS command not configured")

        # Security: basic validation
        if ";" in exe or "|" in exe:
            raise ValueError("Unsafe characters in LAMMPS command")

        parts = shlex.split(exe)
        parts.extend(["-in", f"{uid}.in", "-log", f"{uid}.log"])
        return parts
