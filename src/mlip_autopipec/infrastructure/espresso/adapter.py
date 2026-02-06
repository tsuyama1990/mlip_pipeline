import logging
import tempfile
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import kpts2mp
from ase.calculators.espresso import Espresso
from ase.io import iread, write

from mlip_autopipec.config import OracleConfig
from mlip_autopipec.constants import EXTXYZ_FORMAT
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.infrastructure.espresso.recovery import RecoveryStrategy
from mlip_autopipec.interfaces.oracle import BaseOracle

logger = logging.getLogger(__name__)


class EspressoOracle(BaseOracle):
    def __init__(self, config: OracleConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir

        # Validate work_dir
        if self.work_dir.exists() and not self.work_dir.is_dir():
            msg = f"Work directory path exists but is not a directory: {self.work_dir}"
            raise NotADirectoryError(msg)

        self.recovery_strategy = RecoveryStrategy(config.recovery_recipes)

    def label(self, dataset: Dataset) -> Dataset:
        """
        Labels the structures in the dataset using QE.
        Stream-processes structures to avoid memory issues.
        """
        if not dataset.file_path.exists():
            msg = f"Dataset file not found: {dataset.file_path}"
            raise FileNotFoundError(msg)

        output_path = self.work_dir / f"labeled_{dataset.file_path.name}"

        # Ensure work dir exists
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured work directory exists: {self.work_dir}")
        except OSError as e:
            msg = f"Failed to create work directory {self.work_dir}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

        # Clear output file if exists (we append)
        if output_path.exists():
            output_path.unlink()
            logger.debug(f"Removed existing output file: {output_path}")

        # We reuse a single temporary directory for the entire labeling session
        # to avoid overhead of creating/deleting 1000s of dirs.
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp_str:
            tmp_dir = Path(tmp_str)
            logger.debug(f"Created temporary directory for calculations: {tmp_dir}")

            # Using iread for streaming
            # iread returns a generator.
            for i, atoms in enumerate(iread(dataset.file_path)):
                try:
                    if not isinstance(atoms, Atoms):
                        # Should not happen with ase.io.read/iread usually
                        continue

                    self._label_structure(atoms, tmp_dir, i)

                    # Append result immediately
                    write(output_path, atoms, format=EXTXYZ_FORMAT, append=True)

                except Exception:
                    logger.exception(f"Failed to label structure {i}")
                    # Continue to next structure
                    continue

        return Dataset(file_path=output_path)

    def _label_structure(self, atoms: Atoms, tmp_dir: Path, idx: int) -> None:
        """
        Labels a single structure with retry logic.

        Args:
            atoms (Atoms): The atomic structure to label.
            tmp_dir (Path): The temporary directory for calculation files.
            idx (int): The index of the structure in the dataset (for labeling).
        """
        # Calculate kpts from kspacing
        # kpts2mp requires atoms object to have cell
        kpts = kpts2mp(atoms, self.config.kspacing)  # type: ignore[no-untyped-call]
        logger.debug(f"Calculated k-points for structure {idx}: {kpts}")

        # Base parameters
        base_params = {
            "pseudopotentials": self.config.pseudopotentials,
            "tprnfor": True,
            "tstress": True,
            "kpts": kpts,
            "directory": str(tmp_dir),
            "label": f"calc_{idx}",  # Unique label to avoid file collision
            "command": self.config.command,
            "pseudo_dir": str(self.config.pseudo_dir),
            **self.config.scf_params,
        }

        # Retry loop
        # max_attempts includes initial attempt (1) + retries
        max_attempts = len(self.recovery_strategy.recipes) + 1

        for attempt in range(1, max_attempts + 1):
            try:
                # If first attempt, use base params. If retry, get recovery params.
                current_params = base_params.copy()
                if attempt > 1:
                    logger.debug(f"Attempt {attempt}: retrieving recovery parameters.")
                    current_params = self.recovery_strategy.suggest_next_params(
                        attempt - 1, current_params
                    )

                # Create calculator
                calc = Espresso(**current_params)  # type: ignore[no-untyped-call]
                atoms.calc = calc

                # Trigger calculation
                atoms.get_potential_energy()  # type: ignore[no-untyped-call]
                atoms.get_forces()  # type: ignore[no-untyped-call]
                atoms.get_stress()  # type: ignore[no-untyped-call]

                logger.debug(f"Calculation successful for structure {idx} on attempt {attempt}")

            except Exception as e:
                logger.warning(
                    f"Calculation attempt {attempt} failed for structure {idx}: {e}"
                )
                if attempt == max_attempts:
                    msg = f"All {max_attempts} attempts failed for structure {idx}"
                    # Raise exception to be caught by the loop in label()
                    raise RuntimeError(msg) from e
            else:
                # Success
                return
