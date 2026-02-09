from typing import Iterator, List, Dict, Any, Optional
from pathlib import Path

from mlip_autopipec.components.base import (
    BaseComponent, BaseGenerator, BaseOracle, BaseTrainer, BaseDynamics, BaseValidator
)

# Dummy structure for now
class Structure:
    pass

class MockGenerator(BaseGenerator):
    def generate(self, n_structures: int, cycle: int = 0, metrics: Optional[Dict[str, Any]] = None) -> Iterator[Structure]:
        for _ in range(n_structures):
            yield Structure()

    def enhance(self, structure: Structure) -> Iterator[Structure]:
        yield structure

class MockOracle(BaseOracle):
    def compute(self, structure: Structure) -> Structure:
        return structure

    def compute_batch(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        return structures

class MockTrainer(BaseTrainer):
    def train(self, dataset: List[Structure], initial_potential=None):
        return "mock_potential.yace"

class MockDynamics(BaseDynamics):
    def explore(self, potential: Any, initial_structure: Any, cycle: int = 0, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"halted": False}

class MockValidator(BaseValidator):
    def validate(self, potential: Any) -> Dict[str, Any]:
        return {"passed": True}
