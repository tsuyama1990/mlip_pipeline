import numpy as np
from pathlib import Path
from ase import Atoms
from typing import List, Optional

from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator
from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult

class MockExplorer(BaseExplorer):
    def explore(self, current_potential_path: Path, dataset: Dataset) -> Dataset:
        # Generate 2 dummy structures
        new_structures = []
        for _ in range(2):
            atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
            # Perturb positions slightly to make them unique
            atoms.positions += np.random.rand(3, 3) * 0.1
            new_structures.append(StructureMetadata(structure=atoms, iteration=0))
        return Dataset(structures=new_structures)

class MockOracle(BaseOracle):
    def label(self, dataset: Dataset) -> Dataset:
        # Add random energy and forces to each structure
        for meta in dataset.structures:
            # Using random values
            meta.energy = np.random.uniform(-100, -10)
            # Dimensions of forces array: (N, 3)
            n_atoms = len(meta.structure)
            meta.forces = np.random.uniform(-1, 1, size=(n_atoms, 3)).tolist()
            meta.virial = np.random.uniform(-0.1, 0.1, size=(3, 3)).tolist()
        return dataset

class MockTrainer(BaseTrainer):
    def train(self, train_dataset: Dataset, validation_dataset: Dataset) -> Path:
        # Create a dummy potential file
        potential_path = Path("mock_potential.yace")
        try:
            potential_path.write_text("Mock Potential Data")
        except OSError as e:
            # Audit fix: Error handling
            msg = f"Failed to write mock potential file: {e}"
            raise RuntimeError(msg) from e
        return potential_path

class MockValidator(BaseValidator):
    def validate(self, potential_path: Path) -> ValidationResult:
        return ValidationResult(
            metrics={"rmse_energy": 0.05, "rmse_forces": 0.1},
            is_stable=True
        )
