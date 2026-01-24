import logging
import shlex
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from pathlib import Path
from uuid import uuid4

from ase import Atoms
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTInputParams, DFTResult
from mlip_autopipec.dft.inputs import InputGenerator
from mlip_autopipec.dft.parsers import QEOutputParser
from mlip_autopipec.dft.recovery import RecoveryHandler


class DFTFatalError(Exception):
    pass


class DFTRetriableError(Exception):
    """Exception raised for errors that might be resolved by retrying (e.g. system glitches)."""


logger = logging.getLogger(__name__)


class DFTRunner(ABC):
    """
    Abstract base class for DFT runners.
    """

    @abstractmethod
    def run(self, atoms: Atoms, uid: str | None = None) -> DFTResult:
        """Runs the DFT calculation."""

    @abstractmethod
    def run_batch(self, atoms_iterable: Iterable[Atoms]) -> Generator[DFTResult, None, None]:
        """Runs a batch of DFT calculations."""


class QERunner(DFTRunner):
    """
    Orchestrates Quantum Espresso calculations with auto-recovery and efficient retries.
    """

    INPUT_FILE = "pw.in"
    OUTPUT_FILE = "pw.out"

    def __init__(self, config: DFTConfig, parser_class: type[QEOutputParser] = QEOutputParser):
        """
        Initialize QERunner.

        Args:
            config: DFT Configuration.
            parser_class: Class to use for parsing output (Dependency Injection).
        """
        self.config = config
        self.parser_class = parser_class

    def _validate_command(self, command: str) -> list[str]:
        """
        Validates and splits the command string safely.
        Enforces strict security checks.
        """
        if not command:
            raise DFTFatalError("Command is empty.")

        # Check for forbidden characters that might indicate shell injection attempts
        # independent of shlex splitting.
        forbidden = [";", "&", "|", "`", "$", "(", ")", "<", ">"]
        if any(char in command for char in forbidden):
            raise DFTFatalError("Command contains unsafe shell characters.")

        try:
            parts = shlex.split(command)
        except ValueError as e:
            raise DFTFatalError(f"Command string could not be parsed: {e}") from e

        if not parts:
            raise DFTFatalError("Command parses to empty list.")

        executable = parts[0]

        # Verify executable exists
        if not shutil.which(executable):
            raise DFTFatalError(f"Executable '{executable}' not found in PATH.")

        return parts

    # Use tenacity for exponential backoff on retriable system errors
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(DFTRetriableError),
        reraise=True,
    )
    def _execute_subprocess_with_retry(
        self, cmd: list[str], cwd: Path, stdout_f, timeout: float
    ) -> subprocess.CompletedProcess:
        try:
            # STRICT SECURITY: shell=False is mandatory.
            return subprocess.run(
                cmd,
                check=False,
                shell=False,
                cwd=str(cwd),
                stdout=stdout_f,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
        except OSError as e:
            # OS errors might be transient (e.g. file system blips), so we retry
            raise DFTRetriableError(f"OS Error: {e}") from e

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
            kspacing=self.config.kspacing,
        )

        attempt = 0
        last_error = None

        # Logic retry loop (for physics errors/convergence)
        while attempt <= self.config.max_retries:
            attempt += 1

            with tempfile.TemporaryDirectory(prefix=f"dft_run_{uid}_") as tmpdir:
                work_dir = Path(tmpdir)
                input_str = InputGenerator.create_input_string(atoms, current_params)

                input_path = work_dir / self.INPUT_FILE
                output_path = work_dir / self.OUTPUT_FILE

                input_path.write_text(input_str)
                self._stage_pseudos(work_dir, atoms)

                start_time = time.time()
                full_command = command_parts + ["-in", self.INPUT_FILE]

                returncode = -999
                stdout_content = ""
                stderr_content = ""

                try:
                    with output_path.open("w") as stdout_f:
                        # Use internal retry for system stability
                        proc = self._execute_subprocess_with_retry(
                            full_command, work_dir, stdout_f, self.config.timeout
                        )

                    returncode = proc.returncode
                    stderr_content = proc.stderr if proc.stderr else ""

                    try:
                        stdout_content = output_path.read_text(encoding="utf-8", errors="replace")
                    except FileNotFoundError:
                        stdout_content = ""

                except subprocess.TimeoutExpired as e:
                    logger.error(f"DFT Timeout for job {uid}: {e}")
                    returncode = -1
                    stderr_content = "Timeout Expired"
                except DFTRetriableError as e:
                    logger.error(f"System error persisted after retries for job {uid}: {e}")
                    last_error = e
                    returncode = -998
                    stderr_content = str(e)
                except Exception as e:
                    logger.error(f"Unexpected execution failure for job {uid}: {e}", exc_info=True)
                    last_error = e
                    returncode = -999
                    stderr_content = str(e)

                wall_time = time.time() - start_time

                if returncode != 0:
                    logger.warning(
                        f"QE process exited with code {returncode}. Stderr: {stderr_content[:200]}"
                    )

                if returncode == 0:
                    try:
                        result = self._parse_output(
                            output_path, uid, wall_time, current_params.model_dump(), atoms
                        )
                        if result.succeeded:
                            return result
                    except Exception as e:
                        last_error = e
                        logger.warning(
                            f"Parsing failed despite return code 0 for job {uid}: {e}",
                            exc_info=True,
                        )

                # Physics Error Analysis & Recovery Strategy
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

                    logger.info(
                        f"Retrying job {uid} (Attempt {attempt + 1}) with new params: {new_params_dict}"
                    )
                    continue
                except Exception as e:
                    logger.error(f"Recovery strategy failed for job {uid}: {e}", exc_info=True)
                    last_error = e
                    break

        logger.critical(f"Job {uid} failed completely after {attempt} attempts.")
        raise DFTFatalError(f"Job {uid} failed after {attempt} attempts. Last error: {last_error}")

    def run_batch(self, atoms_iterable: Iterable[Atoms]) -> Generator[DFTResult, None, None]:
        """
        Processes a batch of atoms by consuming an iterable/generator.
        This enables processing large datasets without pre-loading them.
        This method runs sequentially. For parallelism, use TaskQueue in orchestration.
        """
        for atoms in atoms_iterable:
            try:
                # Assuming atoms has info['id'] or generating a new UID
                uid = atoms.info.get("id", str(uuid4()))
                yield self.run(atoms, uid=str(uid))
            except Exception as e:
                logger.error(f"Failed to run DFT for structure {uid}: {e}")
                # We yield a failed result or skip, depending on requirement.
                # Here we skip but log error to keep the stream alive.
                continue

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
        Parses pw.out using the injected parser class.
        """
        parser = self.parser_class()
        return parser.parse(output_path, uid, wall_time, params)
