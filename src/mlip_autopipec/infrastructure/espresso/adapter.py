import logging
import shlex
import tempfile
import uuid
from typing import Any

from ase import Atoms
from ase.calculators.calculator import CalculatorError
from ase.calculators.espresso import Espresso
from ase.io import iread, write

from mlip_autopipec.config.config_model import OracleConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.infrastructure.espresso.recovery import RecoveryStrategy
from mlip_autopipec.interfaces import BaseOracle

logger = logging.getLogger(__name__)


class EspressoOracle(BaseOracle):
    """
    Oracle implementation using Quantum Espresso via ASE.
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config
        if self.config.type != "espresso":
            msg = f"Invalid config type {self.config.type} for EspressoOracle"
            raise ValueError(msg)

        # Ensure pseudo_dir exists if not just checking validity
        if self.config.pseudo_dir and not self.config.pseudo_dir.exists():
            logger.warning(f"Pseudo dir {self.config.pseudo_dir} does not exist locally.")

        # Security check for command
        if self.config.command:
            self._validate_command(self.config.command)

    def _validate_command(self, command: str) -> None:
        """
        Validates the command to prevent injection using a whitelist approach.
        Allowed executables: mpirun, srun, pw.x
        Allowed flags: -np, -n, --ntasks, etc. (we allow alphanumeric and dashes)
        """
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            msg = f"Invalid command format: {e}"
            raise ValueError(msg) from e

        if not tokens:
            msg = "Command cannot be empty"
            raise ValueError(msg)

        executable = tokens[0]
        # Whitelist of safe executables
        # We assume the user provides the full path or it's in PATH
        # We check the basename of the executable
        exe_name = executable.split("/")[-1]

        allowed_executables = {"mpirun", "srun", "pw.x", "mpiexec"}
        # If it's not a known parallel wrapper, it MUST be pw.x (maybe absolute path)
        if exe_name not in allowed_executables and "pw.x" not in exe_name:
            msg = f"Command executable '{exe_name}' is not in the allowed list: {allowed_executables} or 'pw.x'"
            raise ValueError(msg)

        # Check arguments
        # We allow alphanumeric, dashes, equals, slashes, dots, underscores
        # We explicitly disallow shell metacharacters that shlex might have parsed but are risky if executed in shell=True (though we generally avoid shell=True)
        # ASE uses Popen, but sometimes with shell=True depending on implementation.
        # Actually ASE Espresso calculator uses `subprocess.call` or `Popen` with `shell=True` often to handle redirection > espresso.pwo
        # So strictly speaking, injection IS possible if we don't sanitize.

        # We re-verify the full string for dangerous characters just in case shlex missed something subtle
        # or if it's reconstructed.
        dangerous_chars = [";", "|", "&", ">", "<", "$", "`", "\n", "\r", "(", ")"]
        if any(char in command for char in dangerous_chars):
            msg = f"Command contains forbidden characters: {command}"
            raise ValueError(msg)

    def label(self, dataset: Dataset) -> Dataset:
        """
        Calculates Energy, Forces, and Stress for structures in the dataset.
        Returns a new Dataset containing labeled structures.
        """
        output_filename = f"labeled_{uuid.uuid4().hex}.xyz"
        output_path = dataset.file_path.parent / output_filename

        logger.info(f"Starting DFT labeling. Input: {dataset.file_path}, Output: {output_path}")

        count = 0
        success_count = 0
        buffer: list[Atoms] = []
        BUFFER_SIZE = 10

        if not dataset.file_path.exists():
            msg = f"Dataset file not found: {dataset.file_path}"
            raise FileNotFoundError(msg)

        # Create output file (empty)
        output_path.touch()

        # Use a single temporary directory for the batch to minimize IO
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Iterate without loading all into memory
                # Remove index=":" to allow true lazy loading
                for atoms in iread(dataset.file_path):
                    count += 1
                    try:
                        self._run_calculation(atoms, temp_dir)
                        buffer.append(atoms)
                        success_count += 1

                        if len(buffer) >= BUFFER_SIZE:
                            write(output_path, buffer, append=True)
                            buffer.clear()

                    except Exception:
                        logger.exception(f"Failed to label structure {count}")
                        continue

                # Write remaining buffer
                if buffer:
                    write(output_path, buffer, append=True)
                    buffer.clear()

            except Exception:
                logger.exception(f"Error reading dataset file {dataset.file_path}")
                raise

        logger.info(f"Labeled {success_count}/{count} structures. Saved to {output_path}")

        return Dataset(file_path=output_path)

    def _run_calculation(self, atoms: Atoms, work_dir: str) -> None:
        """
        Runs the DFT calculation with self-healing.
        Updates atoms with energy, forces, and stress.
        """
        if not self.config.command or not self.config.pseudo_dir:
            msg = "Missing command or pseudo_dir"
            raise ValueError(msg)

        base_params: dict[str, Any] = {
            "command": self.config.command,
            "pseudo_dir": str(self.config.pseudo_dir),
            "pseudopotentials": self.config.pseudopotentials,
            "kspacing": self.config.kspacing,
            "tprnfor": True,
            "tstress": True,
            "directory": work_dir,  # Reuse temp dir
        }
        base_params.update(self.config.scf_params)

        strategy = RecoveryStrategy(base_params)
        last_error: Exception | None = None

        for params in strategy.iter_attempts():
            try:
                # Create fresh calculator
                calc = Espresso(**params)  # type: ignore[no-untyped-call]
                atoms.calc = calc

                # Trigger calculation
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                atoms.get_forces()  # type: ignore[no-untyped-call]

                # We could delete the .pwi/.pwo files here to save space in the temp dir

            except CalculatorError as e:
                logger.warning(f"DFT calculation failed: {e}. Retrying...")
                last_error = e
                atoms.calc = None
            except Exception:
                logger.exception("Unexpected error in DFT")
                raise
            else:
                return

        msg = "All DFT recovery attempts failed."
        if last_error:
            raise CalculatorError(msg) from last_error
        raise CalculatorError(msg)
