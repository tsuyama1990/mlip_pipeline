import contextlib
import logging
import shlex
import shutil
import subprocess
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.domain_models.inference_models import InferenceResult
from mlip_autopipec.inference.inputs import ScriptGenerator
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
            msg = f"Expected ase.Atoms object, got {type(atoms)}"
            raise TypeError(msg)

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
            # Check for halt in logs first
            max_gamma, halted, step = LammpsLogParser.parse(log_file)

            # If failed (non-zero return) AND not halted, it's a real crash
            if process.returncode != 0 and not halted:
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
                    error_message=f"Process exited with code {process.returncode}",
                    max_gamma_observed=max_gamma,
                    halted=halted,
                    halt_step=step,
                )

            # If returncode != 0 but halted, it is a SUCCESSFUL detection of uncertainty.
            # We treat this as succeeded=True, halted=True.

            # Determine uncertain structures
            uncertain_structures = []
            if halted:
                dump_file = self.work_dir / f"{uid}.dump"
                if dump_file.exists():
                    uncertain_structures.append(dump_file)

            return InferenceResult(
                uid=uid,
                succeeded=True,
                max_gamma_observed=max_gamma,
                halted=halted,
                halt_step=step,
                uncertain_structures=uncertain_structures,
                error_message=None,
            )

        except Exception as e:
            logger.exception(f"Unexpected error in LammpsRunner for {uid}")
            return InferenceResult(
                uid=uid,
                succeeded=False,
                error_message=str(e),
                max_gamma_observed=0.0,
                halted=False,
                halt_step=None,
            )

    def _write_inputs(self, atoms: Atoms, potential_path: Path, uid: str) -> None:
        """Writes data file and input script."""
        data_file = self.work_dir / f"{uid}.data"
        # type: ignore[no-untyped-call]
        atoms.write(str(data_file), format="lammps-data")

        # type: ignore[no-untyped-call]
        elements = sorted(set(atoms.get_chemical_symbols()))

        script_gen = ScriptGenerator(self.config)
        dump_file = self.work_dir / f"{uid}.dump"

        content = script_gen.generate(data_file, potential_path, dump_file, elements)

        input_script = self.work_dir / f"{uid}.in"
        input_script.write_text(content)

    def _build_command(self, uid: str) -> list[str]:
        """Builds the execution command."""
        exe = self.config.lammps_executable
        if not exe:
            msg = "LAMMPS executable not configured"
            raise ValueError(msg)

        exe_str = str(exe)

        # Security: basic validation
        if any(c in exe_str for c in [";", "|", "&", "`", "$", "(", ")"]):
             msg = "Unsafe characters in LAMMPS command"
             raise ValueError(msg)

        parts = shlex.split(exe_str)
        executable = parts[0]

        # Verify executable exists
        if not shutil.which(executable):
            msg = f"Executable '{executable}' not found in PATH."
            raise ValueError(msg)

        parts.extend(["-in", f"{uid}.in", "-log", f"{uid}.log"])
        return parts
