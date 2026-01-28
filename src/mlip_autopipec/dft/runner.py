import logging
import shlex
import shutil
import subprocess
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any, Tuple

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.dft.parsers import BaseDFTParser, QEOutputParser
from mlip_autopipec.dft.recovery import DFTRetriableError, RecoveryHandler, DFTErrorType
from mlip_autopipec.domain_models.dft_models import DFTResult

logger = logging.getLogger(__name__)


class DFTFatalError(Exception):
    """Raised when DFT calculation fails non-recoverably."""


class QERunner:
    """
    Executes DFT calculations using Quantum Espresso (pw.x).
    """

    def __init__(
        self,
        config: DFTConfig,
        work_dir: Path,
        parser_class: type[BaseDFTParser] = QEOutputParser,
    ):
        """
        Initialize QERunner.

        Args:
            config: DFT Configuration.
            work_dir: Directory for calculations.
            parser_class: Class to use for parsing output (Dependency Injection).
        """
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.parser_class = parser_class
        self.recovery_handler = RecoveryHandler(self.config)

    def run(self, atoms: Atoms, uid: str | None = None) -> DFTResult:
        """
        Runs the DFT calculation for the given atoms object.
        """
        if uid is None:
            uid = "calc"

        # 0. Validate Command Security
        try:
            self.config.validate_command_security(self.config.command)
        except ValueError as e:
            logger.exception("Security check failed")
            return DFTResult(
                uid=uid,
                succeeded=False,
                converged=False,
                energy=0.0,
                forces=[],
                error_message=f"Command validation failed: {e}",
                wall_time=0.0,
                parameters=self.config.model_dump(),
                stress=None,
                final_mixing_beta=None,
            )

        # 1. Write Input
        input_path = self.work_dir / "pw.in"
        output_path = self.work_dir / "pw.out"

        try:
            self._write_input(atoms, input_path)
        except Exception as e:
            logger.exception("Failed to write input")
            return DFTResult(
                uid=uid,
                succeeded=False,
                converged=False,
                energy=0.0,
                forces=[],
                error_message=f"Input generation failed: {e}",
                wall_time=0.0,
                parameters=self.config.model_dump(),
                stress=None,
                final_mixing_beta=None,
            )

        # 2. Run with Retries
        params = self.config.model_dump()
        attempt = 0
        success = False
        error_msg = ""

        while attempt < self.config.max_retries:
            attempt += 1
            logger.info(f"Starting DFT execution (Attempt {attempt}/{self.config.max_retries})")

            success, error_msg = self._run_command(input_path, output_path)

            if not success:
                # Check if retriable
                error_type = self.recovery_handler.analyze_error(output_path, error_msg)

                if error_type == DFTErrorType.UNKNOWN:
                    logger.error(f"Fatal error detected: {error_msg}")
                    return DFTResult(
                        uid=uid,
                        succeeded=False,
                        converged=False,
                        energy=0.0,
                        forces=[],
                        error_message=f"Fatal execution error: {error_msg}",
                        wall_time=0.0,
                        parameters=params,
                        stress=None,
                        final_mixing_beta=None,
                    )

                # Attempt recovery
                try:
                    logger.warning(f"Recoverable error {error_type} detected. Applying fix...")
                    new_params = self.recovery_handler.get_strategy(error_type, params)
                    params.update(new_params)
                    # Regenerate input with new params
                    self._write_input(atoms, input_path, params)
                    continue
                except Exception as e:
                    logger.exception("Recovery failed")
                    return DFTResult(
                        uid=uid,
                        succeeded=False,
                        converged=False,
                        energy=0.0,
                        forces=[],
                        error_message=f"Recovery failed: {e}",
                        wall_time=0.0,
                        parameters=params,
                        stress=None,
                        final_mixing_beta=None,
                    )

            if success:
                # 3. Parse Output
                return self._parse_output(output_path, uid, 0.0, params, atoms)

        return DFTResult(
            uid=uid,
            succeeded=False,
            converged=False,
            energy=0.0,
            forces=[],
            error_message=f"Failed after {attempt} attempts. Last error: {error_msg}",
            wall_time=0.0,
            parameters=params,
            stress=None,
            final_mixing_beta=None,
        )

    def _validate_command(self, command: str) -> list[str]:
         """
         Validates and splits the command string safely.
         """
         # This logic is also in DFTConfig, but we can double check here
         return shlex.split(command)

    def _write_input(self, atoms: Atoms, path: Path, params: dict[str, Any] | None = None) -> None:
         if params is None:
             params = self.config.model_dump()

         # Construct QE input manually using ASE write with espresso-in format
         # Mapping config params to QE sections
         input_data = {
             'control': {
                 'calculation': 'scf',
                 'restart_mode': 'from_scratch',
                 'pseudo_dir': str(self.config.pseudopotential_dir),
                 'outdir': './',
                 'tprnfor': True,
                 'tstress': True,
             },
             'system': {
                 'ecutwfc': params.get('ecutwfc', 60.0),
                 'smearing': params.get('smearing', 'mv'),
                 'degauss': params.get('degauss', 0.02),
             },
             'electrons': {
                 'diagonalization': params.get('diagonalization', 'david'),
                 'mixing_beta': params.get('mixing_beta', 0.7),
                 'electron_maxstep': params.get('electron_maxstep', 100),
             }
         }

         # type: ignore[no-untyped-call]
         write(
             path,
             atoms,
             format="espresso-in",
             input_data=input_data,
             pseudopotentials=self.config.pseudopotentials,
             kspacing=params.get('kspacing', 0.05),
         )

    def _run_command(self, input_path: Path, output_path: Path) -> Tuple[bool, str]:
         if not self.config.command:
              return False, "Command is empty"

         parts = self._validate_command(self.config.command)

         executable = parts[0]
         if not shutil.which(executable):
             return False, f"Executable '{executable}' not found in PATH."

         # Construct full command: mpirun -np 4 pw.x -in pw.in > pw.out
         # Note: We use shell=False, so redirection > is not available directly.
         # We handle stdout in Python.

         # Note: QE usually takes input via stdin or -in flag. Standard way is stdin < pw.in
         # But subprocess stdin argument handles that.

         try:
             with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
                 result = subprocess.run(
                     parts,
                     stdin=f_in,
                     stdout=f_out,
                     stderr=subprocess.PIPE,
                     text=True,
                     check=False,
                     timeout=self.config.timeout,
                     cwd=self.work_dir
                 )

             if result.returncode != 0:
                 return False, result.stderr

             return True, ""

         except subprocess.TimeoutExpired:
             return False, "Timeout expired"
         except Exception as e:
             return False, str(e)

    def _parse_output(
         self, output_path: Path, uid: str, wall_time: float, params: dict[str, Any], atoms: Atoms
    ) -> DFTResult:
         parser = self.parser_class()
         return parser.parse(output_path, uid, wall_time, params)
