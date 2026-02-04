from typing import Protocol, runtime_checkable

from mlip_autopipec.config.config_model import ExplorationConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult


@runtime_checkable
class Explorer(Protocol):
    def generate_candidates(self, config: ExplorationConfig) -> list[StructureMetadata]: ...


@runtime_checkable
class Oracle(Protocol):
    def compute(self, structures: list[StructureMetadata]) -> list[StructureMetadata]: ...


@runtime_checkable
class Trainer(Protocol):
    def train(
        self, dataset: list[StructureMetadata], initial_potential: Potential | None = None
    ) -> Potential: ...


@runtime_checkable
class Validator(Protocol):
    def validate(self, potential: Potential) -> ValidationResult: ...
