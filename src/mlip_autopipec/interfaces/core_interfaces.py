from typing import Protocol, runtime_checkable

from mlip_autopipec.config.config_model import (
    DFTConfig,
    ExplorationConfig,
    GlobalConfig,
    TrainingConfig,
)
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult


@runtime_checkable
class Explorer(Protocol):
    """Interface for Structure Generator."""
    def generate_candidates(self, config: ExplorationConfig, n_structures: int) -> list[StructureMetadata]:
        """Generates new candidate structures."""
        ...

@runtime_checkable
class Oracle(Protocol):
    """Interface for Ground Truth Calculator (DFT)."""
    def calculate(self, structures: list[StructureMetadata], config: DFTConfig) -> list[StructureMetadata]:
        """Calculates energy, forces, and stresses for structures."""
        ...

@runtime_checkable
class Trainer(Protocol):
    """Interface for Potential Trainer."""
    def train(self, dataset: list[StructureMetadata], config: TrainingConfig) -> str:
        """Trains a potential and returns the path to the potential file."""
        ...

@runtime_checkable
class Validator(Protocol):
    """Interface for Potential Validator."""
    def validate(self, potential_path: str, config: GlobalConfig) -> ValidationResult:
        """Validates the potential against physical criteria."""
        ...
