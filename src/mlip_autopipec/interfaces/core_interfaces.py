from typing import Protocol, runtime_checkable

from mlip_autopipec.config.config_model import DFTConfig, ExplorationConfig, TrainingConfig
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult


@runtime_checkable
class Explorer(Protocol):
    def generate_candidates(self, config: ExplorationConfig) -> list[StructureMetadata]:
        """
        Generates candidate structures for the active learning loop.
        """
        ...


@runtime_checkable
class Oracle(Protocol):
    def calculate(
        self, structures: list[StructureMetadata], config: DFTConfig
    ) -> list[StructureMetadata]:
        """
        Calculates ground-truth properties (energy, forces, stress) for the structures.
        """
        ...


@runtime_checkable
class Trainer(Protocol):
    def train(self, structures: list[StructureMetadata], config: TrainingConfig) -> str:
        """
        Trains the potential using the provided structures.
        Returns the file path to the trained potential.
        """
        ...


@runtime_checkable
class Validator(Protocol):
    def validate(self, potential_path: str) -> ValidationResult:
        """
        Validates the trained potential against physical criteria.
        """
        ...
