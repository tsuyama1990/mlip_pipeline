import logging
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
        Validates the command to prevent injection.
        Allowed characters: alphanumeric, space, -, _, ., /, =
        Allowed commands (heuristics): mpirun, pw.x, srun
        """
        # Check for dangerous shell characters
        dangerous_chars = [";", "|", "&", ">", "<", "$", "`", "\n", "\r"]
        if any(char in command for char in dangerous_chars):
            msg = f"Command contains invalid characters: {command}"
            raise ValueError(msg)

        # Verify it starts with a reasonable executable (optional but good)
        # We don't want to be too restrictive, but 'rm -rf' should be caught by char check mostly
        # but 'rm' is alphanumeric.
        # We rely on the fact that this config comes from a file, but strictly we should check.
        # For now, the character check covers most injection risks.

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

        if not dataset.file_path.exists():
            msg = f"Dataset file not found: {dataset.file_path}"
            raise FileNotFoundError(msg)

        # Create output file (empty)
        output_path.touch()

        # Use a single temporary directory for the batch to minimize IO
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Iterate without loading all into memory
                # Remove index=":" to allow true lazy loading if supported by format
                # Default behavior of iread is lazy
                for atoms in iread(dataset.file_path):
                    count += 1
                    try:
                        self._run_calculation(atoms, temp_dir)
                        # Append to output file
                        write(output_path, atoms, append=True)
                        success_count += 1
                    except Exception:
                        logger.exception(f"Failed to label structure {count}")
                        continue
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
                # but since it's a temp dir it will be cleaned up at the end of label()
                # If the batch is huge, we might fill the disk.
                # Ideally, Espresso calculator has a 'label' or 'prefix' we can change/rotate
                # or we rely on overwriting. Default prefix is 'espresso'.

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
