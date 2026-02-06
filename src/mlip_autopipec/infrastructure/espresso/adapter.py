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
            # We might want to warn or just assume it exists on the execution node
            logger.warning(f"Pseudo dir {self.config.pseudo_dir} does not exist locally.")

    def label(self, dataset: Dataset) -> Dataset:
        """
        Calculates Energy, Forces, and Stress for structures in the dataset.
        Returns a new Dataset containing labeled structures.
        """
        output_filename = f"labeled_{uuid.uuid4().hex}.xyz"
        # We use the dataset file's parent directory for output
        output_path = dataset.file_path.parent / output_filename

        logger.info(f"Starting DFT labeling. Input: {dataset.file_path}, Output: {output_path}")

        count = 0
        success_count = 0

        # iread yields Atoms objects
        # We assume the file exists.
        if not dataset.file_path.exists():
            msg = f"Dataset file not found: {dataset.file_path}"
            raise FileNotFoundError(msg)

        try:
            for atoms in iread(dataset.file_path, index=":"):
                count += 1
                try:
                    self._run_calculation(atoms)
                    # Append to output file
                    write(output_path, atoms, append=True)
                    success_count += 1
                except Exception:
                    logger.exception(f"Failed to label structure {count}")
                    # We skip this structure.
                    continue
        except Exception:
            logger.exception(f"Error reading dataset file {dataset.file_path}")
            # If we can't read the file, we can't proceed.
            raise

        logger.info(f"Labeled {success_count}/{count} structures. Saved to {output_path}")

        if not output_path.exists():
            output_path.touch()

        return Dataset(file_path=output_path)

    def _run_calculation(self, atoms: Atoms) -> None:
        """
        Runs the DFT calculation with self-healing.
        Updates atoms with energy, forces, and stress.
        """
        # Base parameters from config
        if not self.config.command or not self.config.pseudo_dir:
            msg = "Missing command or pseudo_dir"
            raise ValueError(msg)

        # Create a dictionary for ASE parameters
        # Note: ASE Espresso calculator arguments match the Pydantic fields mostly
        base_params: dict[str, Any] = {
            "command": self.config.command,
            "pseudo_dir": str(self.config.pseudo_dir),
            "pseudopotentials": self.config.pseudopotentials,
            "kspacing": self.config.kspacing,
            "tprnfor": True,
            "tstress": True,
        }
        # Add user scf params
        base_params.update(self.config.scf_params)

        # Use a temporary directory for the calculation files to avoid collisions
        # and ensure cleanup.
        with tempfile.TemporaryDirectory() as temp_dir:
            base_params["directory"] = temp_dir

            strategy = RecoveryStrategy(base_params)

            last_error: Exception | None = None

            for params in strategy.iter_attempts():
                try:
                    # Create fresh calculator for each attempt
                    # We create it inside the loop because ASE calculators can be stateful/brittle
                    calc = Espresso(**params)  # type: ignore[no-untyped-call]
                    atoms.calc = calc

                    # Trigger calculation
                    atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                    atoms.get_forces()  # type: ignore[no-untyped-call]

                except CalculatorError as e:
                    logger.warning(f"DFT calculation failed: {e}. Retrying...")
                    last_error = e
                    # Clean up calculator
                    atoms.calc = None
                except Exception:
                    # Other errors (IO, etc) might not be recoverable by changing params
                    logger.exception("Unexpected error in DFT")
                    raise
                else:
                    # If successful, we are done
                    return

            # If we exhausted all attempts
            msg = "All DFT recovery attempts failed."
            if last_error:
                raise CalculatorError(msg) from last_error
            raise CalculatorError(msg)
