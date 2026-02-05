from typing import Any, Protocol, runtime_checkable

from mlip_autopipec.domain_models.dataset import Dataset
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult


@runtime_checkable
class Explorer(Protocol):
    def generate_candidates(self) -> list[StructureMetadata]:
        """Generate new candidate structures."""
        ...

@runtime_checkable
class Oracle(Protocol):
    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Calculate properties (energy, forces) for a batch of structures."""
        ...

@runtime_checkable
class Trainer(Protocol):
    def train(self, dataset: Dataset) -> Any:
        """Train a potential model using the provided dataset."""
        ...

@runtime_checkable
class Validator(Protocol):
    def validate(self, potential: Any) -> ValidationResult:
        """Validate the potential against physical properties."""
        ...
