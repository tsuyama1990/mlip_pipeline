import logging
import shlex
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from ase import Atoms
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTInputParams, DFTResult
from mlip_autopipec.dft.inputs import InputGenerator
from mlip_autopipec.dft.parsers import QEOutputParser
from mlip_autopipec.dft.recovery import RecoveryHandler


class DFTFatalError(Exception):
    pass

class DFTRetriableError(Exception):
    """Exception raised for errors that might be resolved by retrying."""

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

    def __init__(self, config: DFTConfig, parser_class: type[QEOutputParser] = QEOutputParser, work_dir: Path | None = None) -> None:
        """
        Initialize QERunner.
        """
        self.config = config
        self.parser_class = parser_class
        # work_dir added for compatibility with simple runner tests/logic if needed,
        # though main logic uses tempdirs.
        self.work_dir = work_dir if work_dir else Path("_work_dft")
        if self.work_dir:
            self.work_dir.mkdir(parents=True, exist_ok=True)

    def _validate_command(self, command: str) -> list[str]:
        if not command:
            msg = "Command is empty."
            raise DFTFatalError(msg)

        forbidden = [";", "&", "|", "`", "$", "(", ")", "<", ">"]
        if any(char in command for char in forbidden):
             msg = "Command contains unsafe shell characters."
             raise DFTFatalError(msg)

        try:
            parts = shlex.split(command)
        except ValueError as e:
            msg = f"Command string could not be parsed: {e}"
            raise DFTFatalError(msg) from e

        if not parts:
            msg = "Command parses to empty list."
            raise DFTFatalError(msg)

        executable = parts[0]
        if not shutil.which(executable):
             msg = f"Executable '{executable}' not found in PATH."
             raise DFTFatalError(msg)

        return parts

    def _execute_subprocess_with_retry(
        self, cmd: list[str], cwd: Path, stdout_f: Any, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(DFTRetriableError),
            reraise=True,
        ):
            with attempt:
                try:
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
                    msg = f"OS Error: {e}"
                    raise DFTRetriableError(msg) from e
        msg = "Retrying loop failed."
        raise RuntimeError(msg)

    def run(self, atoms: Atoms, uid: str | None = None) -> DFTResult:
        if uid is None:
            uid = str(uuid4())

        command_parts = self._validate_command(self.config.command)

        # Merge configs for params
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

                # Input Generation
                # We use the InputGenerator from dft/inputs.py if available,
                # but fall back to internal _write_input logic if needed to match Simple runner.
                # Here we use InputGenerator as it's more robust.
                try:
                    input_str = InputGenerator.create_input_string(atoms, current_params)
                except Exception as e:
                    return DFTResult(
                        uid=uid, energy=0.0, forces=[], stress=[], succeeded=False,
                        converged=False, error_message=f"Input generation failed: {e}",
                        wall_time=0.0, parameters={}
                    )

                input_path = work_dir / self.INPUT_FILE
                output_path = work_dir / self.OUTPUT_FILE
                input_path.write_text(input_str)

                self._stage_pseudos(work_dir, atoms)

                start_time = time.time()
                full_command = [*command_parts, "-in", self.INPUT_FILE]

                returncode = -999
                stdout_content = ""
                stderr_content = ""

                try:
                    with output_path.open("w") as stdout_f:
                        proc = self._execute_subprocess_with_retry(
                            full_command, work_dir, stdout_f, self.config.timeout
                        )
                    returncode = proc.returncode
                    stderr_content = proc.stderr or ""
                    if output_path.exists():
                        stdout_content = output_path.read_text(encoding="utf-8", errors="replace")

                except subprocess.TimeoutExpired:
                    logger.exception(f"DFT Timeout for job {uid}")
                    returncode = -1
                    stderr_content = "Timeout Expired"
                except Exception as e:
                     logger.exception(f"Execution failure for job {uid}")
                     last_error = e
                     returncode = -999
                     stderr_content = str(e)

                wall_time = time.time() - start_time

                if returncode == 0:
                    try:
                        result = self._parse_output(output_path, uid, wall_time, current_params.model_dump(), atoms)
                        if result.succeeded:
                            return result
                    except Exception:
                        logger.exception(f"Parsing failed despite return code 0 for job {uid}")

                # Recovery
                error_type = RecoveryHandler.analyze(stdout_content, stderr_content)

                if not self.config.recoverable:
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
                    logger.info(f"Retrying job {uid} (Attempt {attempt + 1})")
                except Exception:
                    logger.exception("Recovery strategy failed")
                    break

        msg = f"Job {uid} failed after {attempt} attempts."
        return DFTResult(
            uid=uid, energy=0.0, forces=[], stress=[], succeeded=False,
            converged=False, error_message=f"{msg} Last error: {last_error}",
            wall_time=0.0, parameters={}
        )

    def run_batch(self, atoms_iterable: Iterable[Atoms]) -> Generator[DFTResult, None, None]:
        for atoms in atoms_iterable:
            uid = atoms.info.get("id", str(uuid4()))
            yield self.run(atoms, uid=str(uid))

    def _stage_pseudos(self, work_dir: Path, atoms: Atoms) -> None:
        from mlip_autopipec.dft.constants import SSSP_EFFICIENCY_1_1
        pseudo_src_dir = self.config.pseudopotential_dir
        # type: ignore[no-untyped-call]
        unique_species = set(atoms.get_chemical_symbols())
        for s in unique_species:
            if s in SSSP_EFFICIENCY_1_1:
                u_file = SSSP_EFFICIENCY_1_1[s]
                src = pseudo_src_dir / u_file
                dst = work_dir / u_file
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)

    def _parse_output(
        self, output_path: Path, uid: str, wall_time: float, params: dict[str, Any], atoms: Atoms
    ) -> DFTResult:
        parser = self.parser_class()
        return parser.parse(output_path, uid, wall_time, params)

    # Simple compatibility methods if needed by tests calling internal methods
    def _write_input(self, atoms: Atoms, path: Path) -> None:
        # Fallback to simple write if needed, or redirect to InputGenerator
        params = DFTInputParams(
            ecutwfc=self.config.ecutwfc,
            kspacing=self.config.kspacing
        )
        content = InputGenerator.create_input_string(atoms, params)
        path.write_text(content)
