import logging
import numpy as np
from pathlib import Path
from typing import List, Optional
from ase import Atoms

from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator
from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult

logger = logging.getLogger(__name__)

class MockExplorer(BaseExplorer):
    def explore(self, potential_path: Optional[Path], dataset: Dataset) -> List[StructureMetadata]:
        logger.info("MockExplorer: Exploring...")
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
        meta = StructureMetadata(structure=atoms, source="mock_exploration")
        return [meta]

class MockOracle(BaseOracle):
    def label(self, structures: List[StructureMetadata]) -> List[StructureMetadata]:
        logger.info(f"MockOracle: Labeling {len(structures)} structures...")
        labeled = []
        for s in structures:
            s.energy = -13.6
            s.forces = np.random.rand(len(s.structure), 3).tolist()
            s.virial = np.zeros((3, 3)).tolist()
            labeled.append(s)
        return labeled

class MockTrainer(BaseTrainer):
    def train(self, dataset: Dataset, validation_set: Optional[Dataset] = None) -> Path:
        logger.info("MockTrainer: Training potential...")
        pot_path = Path("mock_potential.yace")
        pot_path.touch()
        return pot_path

class MockValidator(BaseValidator):
    def validate(self, potential_path: Path) -> ValidationResult:
        logger.info("MockValidator: Validating potential...")
        return ValidationResult(passed=True, metrics={"rmse_energy": 0.001, "rmse_force": 0.01}, artifacts={})
