import logging
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTInputParams, DFTResult
from mlip_autopipec.dft.inputs import InputGenerator
from mlip_autopipec.dft.parsers import QEOutputParser
from mlip_autopipec.dft.recovery import RecoveryHandler


class DFTFatalError(Exception):
    pass

logger = logging.getLogger(__name__)

class QERunner:
    """
    Orchestrates Quantum Espresso calculations with auto-recovery.
    """

    def __init__(self, config: DFTConfig):
        self.config = config

    def _validate_command(self, command: str) -> list[str]:
        """
        Validates and splits the command string safely.
        """
        if any(char in command for char in [";", "&", "|", "`", "$"]):
             raise DFTFatalError("Command contains potentially unsafe shell characters.")

        parts = shlex.split(command)
        if not parts:
            raise DFTFatalError("Command is empty.")

        executable = parts[0]
        if not shutil.which(executable):
             raise DFTFatalError(f"Executable '{executable}' not found in PATH.")

        return parts

    def run(self, atoms: Atoms, uid: str | None = None) -> DFTResult:
        """
        Runs the DFT calculation for the given atoms object.
        """
        if uid is None:
            uid = str(uuid4())

        command_parts = self._validate_command(self.config.command)

        current_params = DFTInputParams(
            mixing_beta=self.config.mixing_beta,
            diagonalization=self.config.diagonalization,
            smearing=self.config.smearing,
            degauss=self.config.degauss,
            ecutwfc=self.config.ecutwfc,
            kspacing=self.config.kspacing
        )

        attempt = 0
        last_error = None

        while attempt <= self.config.max_retries:
            attempt += 1

            with tempfile.TemporaryDirectory(prefix=f"dft_run_{uid}_") as tmpdir:
                work_dir = Path(tmpdir)
                input_str = InputGenerator.create_input_string(atoms, current_params)

                input_path = work_dir / "pw.in"
                output_path = work_dir / "pw.out"

                input_path.write_text(input_str)
                self._stage_pseudos(work_dir, atoms)

                start_time = time.time()
                full_command = command_parts + ["-in", "pw.in"]

                try:
                    with output_path.open("w") as stdout_f:
                        proc = subprocess.run(
                            full_command,
                            check=False,
                            shell=False,
                            cwd=str(work_dir),
                            stdout=stdout_f,
                            stderr=subprocess.PIPE,
                            timeout=self.config.timeout,
                            text=True,
                        )

                    returncode = proc.returncode

                    try:
                        stdout_content = output_path.read_text(encoding="utf-8", errors="replace")
                    except FileNotFoundError:
                        stdout_content = ""
                    stderr_content = proc.stderr if proc.stderr else ""

                except subprocess.TimeoutExpired as e:
                    logger.error(f"DFT Timeout for job {uid}: {e}")
                    returncode = -1
                    try:
                        stdout_content = output_path.read_text(encoding="utf-8", errors="replace")
                    except FileNotFoundError:
                         stdout_content = ""
                    stderr_content = "Timeout Expired"
                except OSError as e:
                     logger.error(f"OS Error executing subprocess for job {uid}: {e}", exc_info=True)
                     last_error = e
                     returncode = -998
                     stdout_content = ""
                     stderr_content = str(e)
                except Exception as e:
                     logger.error(f"Subprocess execution failed for job {uid}: {e}", exc_info=True)
                     last_error = e
                     returncode = -999
                     stdout_content = ""
                     stderr_content = str(e)

                wall_time = time.time() - start_time

                if returncode != 0:
                    logger.warning(f"QE process exited with code {returncode}. Stderr: {stderr_content[:200]}")

                if returncode == 0:
                    try:
                        result = self._parse_output(output_path, uid, wall_time, current_params.model_dump(), atoms)
                        if result.succeeded:
                            return result
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Parsing failed despite return code 0 for job {uid}: {e}", exc_info=True)

                error_type = RecoveryHandler.analyze(stdout_content, stderr_content)

                if not self.config.recoverable or attempt > self.config.max_retries:
                    break

                if error_type.name == "NONE" and returncode != 0:
                     msg = f"Process exited with {returncode} but no known error pattern found."
                     logger.error(msg)
                     last_error = DFTFatalError(msg)
                     break

                try:
                    current_params_dict = current_params.model_dump()
                    new_params_dict = RecoveryHandler.get_strategy(error_type, current_params_dict)
                    current_params = DFTInputParams(**new_params_dict)

                    logger.info(f"Retrying job {uid} (Attempt {attempt + 1}) with new params: {new_params_dict}")
                    continue
                except Exception as e:
                    logger.error(f"Recovery strategy failed for job {uid}: {e}", exc_info=True)
                    last_error = e
                    break

        logger.critical(f"Job {uid} failed completely after {attempt} attempts.")
        raise DFTFatalError(f"Job {uid} failed after {attempt} attempts. Last error: {last_error}")

    def _stage_pseudos(self, work_dir: Path, atoms: Atoms):
        """
        Symlinks required pseudopotentials to the working directory.
        """
        from mlip_autopipec.dft.constants import SSSP_EFFICIENCY_1_1

        pseudo_src_dir = self.config.pseudopotential_dir

        unique_species = set(atoms.get_chemical_symbols())
        for s in unique_species:
            if s in SSSP_EFFICIENCY_1_1:
                u_file = SSSP_EFFICIENCY_1_1[s]
                src = pseudo_src_dir / u_file
                dst = work_dir / u_file
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)

    def _parse_output(
        self, output_path: Path, uid: str, wall_time: float, params: dict, atoms: Atoms
    ) -> DFTResult:
        """
        Parses pw.out using QEOutputParser.
        """
        parser = QEOutputParser()
        return parser.parse(output_path, uid, wall_time, params)
