import logging
from pathlib import Path

import numpy as np
from ase import Atoms

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

        Args:
            current_potential_path: Path to the current potential (unused).
            dataset: Current dataset (unused).

        Returns:
            Dataset containing new candidate structures.
        """
        logger.info("MockExplorer: Generating new candidates...")
        # Generate 2 dummy structures
        new_structures = []
        for _ in range(2):
            atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
            # Perturb positions slightly to make them unique
            atoms.positions += np.random.rand(3, 3) * 0.1
            new_structures.append(StructureMetadata(structure=atoms, iteration=0))

        logger.info(f"MockExplorer: Generated {len(new_structures)} structures.")
        return Dataset(structures=new_structures)

class MockOracle(BaseOracle):
    """
    Mock implementation of an Oracle that assigns random energies and forces.
    """
    def label(self, dataset: Dataset) -> Dataset:
        """
        Labels the dataset with random physical properties.

        Args:
            dataset: Dataset containing structures to label.

        Returns:
            Labeled Dataset with energy, forces, and virial.
        """
        logger.info(f"MockOracle: Labeling {len(dataset.structures)} structures...")
        # Add random energy and forces to each structure
        for meta in dataset.structures:
            if meta.structure is None:
                logger.warning("MockOracle: Encountered StructureMetadata with None structure.")
                continue

            # Using random values
            meta.energy = np.random.uniform(-100, -10)

            # Dimensions of forces array: (N, 3)
            n_atoms = len(meta.structure)
            meta.forces = np.random.uniform(-1, 1, size=(n_atoms, 3)).tolist()
            meta.virial = np.random.uniform(-0.1, 0.1, size=(3, 3)).tolist()

        logger.info("MockOracle: Labeling complete.")
        return dataset

class MockTrainer(BaseTrainer):
    """
    Mock implementation of a Trainer that produces a dummy potential file.
    """
    def train(self, train_dataset: Dataset, validation_dataset: Dataset) -> Path:
        """
        Simulates training a potential.

        Args:
            train_dataset: Dataset for training.
            validation_dataset: Dataset for validation.

        Returns:
            Path to the trained potential file.
        """
        logger.info("MockTrainer: Starting training...")
        # Create a dummy potential file
        potential_path = Path("mock_potential.yace")
        try:
            potential_path.write_text("Mock Potential Data")
            logger.info(f"MockTrainer: Wrote potential to {potential_path}")
        except OSError as e:
            # Audit fix: Error handling
            msg = f"Failed to write mock potential file: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        return potential_path

class MockValidator(BaseValidator):
    """
    Mock implementation of a Validator that returns fixed metrics.
    """
    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Simulates validation of a potential.

        Args:
            potential_path: Path to the potential to validate.

        Returns:
            ValidationResult containing fixed metrics.
        """
        logger.info(f"MockValidator: Validating potential at {potential_path}")
        return ValidationResult(
            metrics={"rmse_energy": 0.05, "rmse_forces": 0.1},
            is_stable=True
        )
