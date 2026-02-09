from typing import Iterator, List, Dict, Any, Optional
from pathlib import Path

from mlip_autopipec.components.base import (
    BaseComponent, BaseGenerator, BaseOracle, BaseTrainer, BaseDynamics, BaseValidator
)
from mlip_autopipec.domain_models.datastructures import Dataset, Structure
from mlip_autopipec.constants import EXT_POTENTIAL

class MockStructure(Structure):
    """Mock structure class for testing."""
    def __init__(self, atoms: int = 1):
        super().__init__(data={"atoms": atoms})
        self.atoms = atoms

    def __repr__(self) -> str:
        return f"MockStructure(atoms={self.atoms})"

class MockGenerator(BaseGenerator):
    """Generates mock structures."""

    def generate(self, n_structures: int, cycle: int = 0, metrics: Optional[Dict[str, Any]] = None) -> Iterator[Structure]:
        for i in range(n_structures):
            yield MockStructure(atoms=i)

    def enhance(self, structure: Structure) -> Iterator[Structure]:
        if not isinstance(structure, MockStructure):
            # In real scenario, we might accept generic Structure, but for mock testing we want strictness
            pass
        yield structure

class MockOracle(BaseOracle):
    """Mock oracle."""

    def compute(self, structure: Structure) -> Structure:
        return structure

    def compute_batch(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        for s in structures:
            yield self.compute(s)

class MockTrainer(BaseTrainer):
    """Mock trainer."""

    def train(self, dataset: Dataset, initial_potential: Any = None) -> str:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset, got {type(dataset)}")
        return f"mock_potential{EXT_POTENTIAL}"

class MockDynamics(BaseDynamics):
    """Mock dynamics."""

    def explore(self, potential: Any, initial_structure: Structure, cycle: int = 0, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"halted": False}

class MockValidator(BaseValidator):
    """Mock validator."""

    def validate(self, potential: Any) -> Dict[str, Any]:
        return {"passed": True}
