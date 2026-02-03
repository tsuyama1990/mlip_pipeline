from typing import Protocol, runtime_checkable

from mlip_autopipec.domain_models.dynamics import MDState
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult


@runtime_checkable
class Explorer(Protocol):
    def explore(self, state: MDState | None = None) -> list[StructureMetadata]:
        """Generate new structures for exploration."""
        ...


@runtime_checkable
class Oracle(Protocol):
    def compute(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Compute properties (DFT) for the given structures."""
        ...


@runtime_checkable
class Trainer(Protocol):
    def train(self, structures: list[StructureMetadata]) -> str:
        """Train a potential using the given structures."""
        ...


@runtime_checkable
class Validator(Protocol):
    def validate(self, potential_path: str) -> ValidationResult:
        """Validate the trained potential."""
        ...
