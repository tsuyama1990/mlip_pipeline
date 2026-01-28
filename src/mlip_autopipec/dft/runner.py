import logging
import shlex
import shutil
import subprocess
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.parsers import BaseDFTParser, QEOutputParser
from mlip_autopipec.dft.recovery import DFTRetriableError, RecoveryHandler

logger = logging.getLogger(__name__)


class DFTFatalError(Exception):
    """Raised when DFT calculation fails non-recoverably."""


class QERunner:
    def __init__(
        self, config: DFTConfig, work_dir: Path, parser_class: type[BaseDFTParser] = QEOutputParser
    ):
        """
        Args:
            config: DFT Configuration object.
            work_dir: Directory where calculations will be run.
            parser_class: Class to use for parsing output (Dependency Injection).
        """
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.parser_class = parser_class

    def run(self, atoms: Atoms, uid: str | None = None) -> DFTResult:
        """
        Runs the DFT calculation for the given atoms object.
        """
        if uid is None:
            uid = "dft_calc"

        # 1. Write Input
        input_path = self.work_dir / "pw.in"
        output_path = self.work_dir / "pw.out"

        self._validate_command(self.config.command)

        try:
            self._write_input(atoms, input_path)
        except Exception as e:
            return DFTResult(
                energy=0.0,
                forces=[],
                converged=False,
                error_message=f"Input generation failed: {e}",
            )

        # 2. Run with Retries
        recovery = RecoveryHandler(self.config)
        params = self.config.model_dump()

        # Symlink pseudopotentials
        self._stage_pseudos(self.work_dir, atoms)

        attempt = 0
        while attempt <= self.config.max_retries:
            attempt += 1
            success, error_msg = self._run_command(input_path, output_path)

            if success:
                # 3. Parse Output
                # Calculate walltime if possible, for now 0.0
                return self._parse_output(output_path, uid, 0.0, params, atoms)

            # Handle Failure
            logger.warning(f"DFT Attempt {attempt} failed: {error_msg}")

            # Parse output for specific error
            error_type = recovery.analyze_error(output_path, error_msg)

            if not self.config.recoverable:
                break

            try:
                new_params = recovery.get_strategy(error_type, params)
                logger.info(f"Applying recovery strategy: {new_params}")
                params.update(new_params)
                # Re-write input with new params
                # Note: We need to update self.config or pass params to _write_input
                # _write_input currently uses self.config.
                # Refactoring _write_input to accept params override
                self._write_input(atoms, input_path, params_override=params)

            except DFTRetriableError:
                continue  # Retry with same params? No, that's infinite loop.
                # If get_strategy raises DFTRetriableError, it means we can't recover
                break
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                break

        return DFTResult(
            energy=0.0,
            forces=[],
            converged=False,
            error_message=f"Failed after {attempt} attempts. Last error: {error_msg}",
        )

    def _validate_command(self, command: str) -> list[str]:
        """
        Validates and splits the command string safely.
        """
        forbidden = [";", "&", "|", "`", "$", "(", ")"]
        if any(char in command for char in forbidden):
            raise DFTFatalError("Command contains unsafe shell characters.")

        try:
            parts = shlex.split(command)
            if not parts:
                raise DFTFatalError("Command is empty")
            return parts
        except ValueError as e:
            raise DFTFatalError(f"Invalid command string: {e}") from e

    def _run_command(self, input_path: Path, output_path: Path) -> tuple[bool, str]:
        if not self.config.command:
            return False, "Command is empty"

        parts = self._validate_command(self.config.command)
        executable = parts[0]

        # Verify executable exists
        if not shutil.which(executable):
            raise DFTFatalError(f"Executable '{executable}' not found in PATH.")

        # Prepare command
        cmd = parts + [
            "-in",
            str(input_path.name),
        ]  # standard QE usage often is `pw.x < pw.in > pw.out` or `pw.x -in pw.in`
        # But QE standard is often `pw.x -input pw.in` or stdin redirection.
        # Let's assume standard redirection logic is handled by caller or we use stdin/stdout

        # Using stdin/stdout for QE
        try:
            with open(input_path) as f_in, open(output_path, "w") as f_out:
                subprocess.run(
                    parts,
                    stdin=f_in,
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    cwd=self.work_dir,
                    check=True,
                    timeout=3600,  # 1 hour timeout hardcoded for now, or from config
                    shell=False,  # SECURITY
                )
            return True, ""
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else "Unknown error"
            return False, stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout Expired"
        except Exception as e:
            return False, str(e)

    def _write_input(self, atoms: Atoms, path: Path, params_override: dict | None = None) -> None:
        # Construct params
        params = self.config.model_dump()
        if params_override:
            params.update(params_override)

        # Helper to extract dict from params
        input_data = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "pseudo_dir": str(self.config.pseudopotential_dir),
                "outdir": "./",
                "tprnfor": True,
                "tstress": True,
            },
            "system": {
                "ecutwfc": params.get("ecutwfc", 60.0),
                "smearing": params.get("smearing", "mv"),
                "degauss": params.get("degauss", 0.02),
            },
            "electrons": {
                "diagonalization": params.get("diagonalization", "david"),
            },
        }

        # Add input_data to atoms object for ase.io.write
        # Or use DFTInputGenerator if we had one connected.
        # For now, relying on ASE's built-in espresso writer via simple kwargs or dict

        # ASE espresso calculator/writer uses 'input_data' dict

        # type: ignore[no-untyped-call]
        write(
            path,
            atoms,
            format="espresso-in",
            input_data=input_data,
            pseudopotentials=self.config.pseudopotentials,
            kspacing=params.get("kspacing", 0.05),
        )

    def _stage_pseudos(self, work_dir: Path, atoms: Atoms):
        """
        Symlinks required pseudopotentials to the working directory.
        """
        if not self.config.pseudopotentials:
            return

        symbols = set(atoms.get_chemical_symbols())
        for sym in symbols:
            if sym in self.config.pseudopotentials:
                fname = self.config.pseudopotentials[sym]
                src = self.config.pseudopotential_dir / fname
                dst = work_dir / fname
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)

    def _parse_output(
        self, output_path: Path, uid: str, wall_time: float, params: dict[str, Any], atoms: Atoms
    ) -> DFTResult:
        parser = self.parser_class()
        return parser.parse(output_path, uid, wall_time, params)

    def run_batch(self, atoms_iterable: Iterable[Atoms]) -> Generator[DFTResult, None, None]:
        """
        Processes a batch of atoms by consuming an iterable/generator.
        """
        for i, atoms in enumerate(atoms_iterable):
            uid = f"job_{i}"
            yield self.run(atoms, uid=uid)
