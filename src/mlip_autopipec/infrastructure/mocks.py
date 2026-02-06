import logging
from collections.abc import Generator
from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.config import TrainerConfig, ValidatorConfig
from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator

logger = logging.getLogger(__name__)

class MockExplorer(BaseExplorer):
    """
    Mock implementation of an Explorer that generates random H2O structures.
    """
    def explore(self, current_potential_path: Path, dataset: Dataset) -> Dataset:
        """
        Generates dummy candidate structures.
        """
        logger.info("MockExplorer: Generating new candidates...")

        def _generate() -> Generator[StructureMetadata, None, None]:
            for _ in range(2):
                atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
                atoms.positions += np.random.rand(3, 3) * 0.1
                yield StructureMetadata(structure=atoms, iteration=0)

        new_structures = list(_generate())
        logger.info(f"MockExplorer: Generated {len(new_structures)} structures.")
        return Dataset(structures=new_structures)

class MockOracle(BaseOracle):
    """
    Mock implementation of an Oracle that assigns random energies and forces.
    """
    def label(self, dataset: Dataset) -> Dataset:
        """
        Labels the dataset with random physical properties.
        """
        logger.info(f"MockOracle: Labeling {len(dataset.structures)} structures...")

        labeled_structures: list[StructureMetadata] = []
        # If dataset comes from file, we should handle it, but for Cycle 01 Mock,
        # we assume Explorer returns in-memory structures.
        if not dataset.structures and dataset.file_path:
             logger.warning("MockOracle: Received empty structure list but file_path is set. Streaming not implemented for MockOracle in Cycle 01.")

        for meta in dataset.structures:
            if meta.structure is None:
                logger.warning("MockOracle: Encountered StructureMetadata with None structure.")
                continue

            # Simulate labeling
            meta.energy = np.random.uniform(-100, -10)
            n_atoms = len(meta.structure)
            meta.forces = np.random.uniform(-1, 1, size=(n_atoms, 3)).tolist()
            meta.virial = np.random.uniform(-0.1, 0.1, size=(3, 3)).tolist()
            labeled_structures.append(meta)

        logger.info("MockOracle: Labeling complete.")
        return Dataset(structures=labeled_structures)

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

        if train_dataset.file_path:
            logger.info(f"MockTrainer: Training from file {train_dataset.file_path}")
        else:
            logger.info(f"MockTrainer: Training from memory ({len(train_dataset.structures)} items)")

        potential_filename = self.config.potential_output_name
        potential_path = Path(potential_filename)

        try:
            potential_path.write_text("Mock Potential Data")
            logger.info(f"MockTrainer: Wrote potential to {potential_path}")
        except OSError as e:
            msg = f"Failed to write mock potential file: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        return potential_path

class MockValidator(BaseValidator):
    """
    Mock implementation of a Validator that returns metrics (randomized or config-based).
    """
    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config

    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Validates the potential by generating random metrics.
        """
        logger.info(f"MockValidator: Validating potential at {potential_path}")
        # Generate random metrics to be more realistic as requested
        rmse_energy = np.random.uniform(0.001, 0.1)
        rmse_forces = np.random.uniform(0.01, 0.5)

        return ValidationResult(
            metrics={"rmse_energy": rmse_energy, "rmse_forces": rmse_forces},
            is_stable=True
        )
