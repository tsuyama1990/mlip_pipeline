import logging
import time
import uuid
from collections.abc import Generator
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from mlip_autopipec.config import ExplorerConfig, TrainerConfig, ValidatorConfig
from mlip_autopipec.domain_models import Dataset, ValidationResult
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator

logger = logging.getLogger(__name__)


class MockExplorer(BaseExplorer):
    """
    Mock implementation of an Explorer that generates random H2O structures.
    """

    def __init__(self, config: ExplorerConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir

    def explore(self, current_potential_path: Path, dataset: Dataset) -> Dataset:
        """
        Generates dummy candidate structures and writes them to disk.
        """
        logger.info("MockExplorer: Generating new candidates...")

        # Generate unique filename for candidates
        candidates_file = self.work_dir / f"candidates_{uuid.uuid4().hex}.xyz"

        def _generate() -> Generator[Atoms, None, None]:
            for _ in range(self.config.n_structures):
                # Type hints for positions as requested
                positions: list[list[float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
                atoms = Atoms("H2O", positions=positions)
                # Perturb positions
                perturbation = np.random.rand(3, 3) * 0.1
                atoms.positions += perturbation
                yield atoms

        atoms_list = list(_generate())
        write(candidates_file, atoms_list)
        logger.info(f"MockExplorer: Generated {len(atoms_list)} structures to {candidates_file}.")

        return Dataset(file_path=candidates_file)


class MockOracle(BaseOracle):
    """
    Mock implementation of an Oracle that assigns random energies and forces.
    """

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def label(self, dataset: Dataset) -> Dataset:
        """
        Labels the dataset with random physical properties.
        Generates random energy (uniform -100 to -10), forces (-1 to 1), and virial (-0.1 to 0.1).
        """
        logger.info(f"MockOracle: Labeling structures from {dataset.file_path}...")

        labeled_file = self.work_dir / f"labeled_{uuid.uuid4().hex}.xyz"
        labeled_atoms = []

        # Read atoms from file
        # Check if file exists
        if not dataset.file_path.exists():
            msg = f"Dataset file {dataset.file_path} does not exist"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            # reading all for mock
            atoms_list = read(dataset.file_path, index=":")
            if isinstance(atoms_list, Atoms):
                atoms_list = [atoms_list]
        except Exception as e:
            msg = f"Failed to read dataset file: {e}"
            logger.exception(msg)
            raise

        for atoms in atoms_list:
            # Simulate labeling
            # Random generation as requested in docstrings
            energy = np.random.uniform(-100, -10)
            n_atoms = len(atoms)
            forces = np.random.uniform(-1, 1, size=(n_atoms, 3))
            virial = np.random.uniform(-0.1, 0.1, size=(3, 3))  # simplified 3x3

            # Set properties on atoms so they are written to extxyz
            atoms.info["energy"] = energy
            # stress in ASE is Voigt order (xx, yy, zz, yz, xz, xy)
            atoms.info["stress"] = np.array(
                [virial[0, 0], virial[1, 1], virial[2, 2], virial[1, 2], virial[0, 2], virial[0, 1]]
            )
            # ASE expects forces in arrays
            atoms.new_array("forces", forces)  # type: ignore[no-untyped-call]

            labeled_atoms.append(atoms)

        write(labeled_file, labeled_atoms)
        logger.info(f"MockOracle: Labeling complete. Wrote to {labeled_file}.")
        return Dataset(file_path=labeled_file)


class MockTrainer(BaseTrainer):
    """
    Mock implementation of a Trainer that produces a dummy potential file.
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def train(self, train_dataset: Dataset, validation_dataset: Dataset) -> Path:
        """
        Simulates training a potential.
        """
        logger.info("MockTrainer: Starting training...")
        logger.info(f"MockTrainer: Training from file {train_dataset.file_path}")

        # Write to the same directory as the dataset to ensure it's in work_dir
        output_dir = train_dataset.file_path.parent
        potential_path = output_dir / self.config.potential_output_name

        content = "Mock Potential Data"

        # Retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                potential_path.write_text(content, encoding="utf-8")
                logger.info(f"MockTrainer: Wrote potential to {potential_path}")
                break
            except OSError as e:
                if attempt == max_retries - 1:
                    msg = f"Failed to write mock potential file after {max_retries} attempts: {e}"
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                logger.warning(f"MockTrainer: Write failed (attempt {attempt + 1}), retrying...")
                time.sleep(0.1)

        return potential_path


class MockValidator(BaseValidator):
    """
    Mock implementation of a Validator that returns metrics (randomized or config-based).
    """

    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config

    def validate(self, potential_path: Path) -> ValidationResult:
        logger.info(f"MockValidator: Validating potential at {potential_path}")
        # Generate random metrics to be more realistic as requested
        rmse_energy = np.random.uniform(0.001, 0.1)
        rmse_forces = np.random.uniform(0.01, 0.5)

        return ValidationResult(
            metrics={"rmse_energy": rmse_energy, "rmse_forces": rmse_forces}, is_stable=True
        )
