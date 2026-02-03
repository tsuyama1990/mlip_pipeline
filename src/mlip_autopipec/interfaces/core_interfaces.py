from pathlib import Path
from typing import Protocol, runtime_checkable

from ase import Atoms

from mlip_autopipec.domain_models.validation import ValidationResult

# Type aliases
Structure = Atoms
LabelledStructure = Atoms
PotentialPath = Path

@runtime_checkable
class Explorer(Protocol):
    def explore(self, current_potential: PotentialPath | None) -> list[Structure]:
        """Generates candidate structures."""
        ...

@runtime_checkable
class Oracle(Protocol):
    def compute(self, structures: list[Structure]) -> list[LabelledStructure]:
        """Calculates energy, forces, and virial for structures."""
        ...

@runtime_checkable
class Trainer(Protocol):
    def train(self, dataset: list[LabelledStructure]) -> PotentialPath:
        """Trains a potential on the given dataset."""
        ...

@runtime_checkable
class Validator(Protocol):
    def validate(self, potential: PotentialPath) -> ValidationResult:
        """Validates the potential against physics metrics."""
        ...
