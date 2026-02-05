from pathlib import Path
from typing import Protocol, runtime_checkable

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult


@runtime_checkable
class Explorer(Protocol):
    def generate(self, config: GlobalConfig) -> list[StructureMetadata]:
        """
        Generates new atomic structures based on the configuration.
        """
        ...


@runtime_checkable
class Oracle(Protocol):
    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """
        Calculates properties (energy, forces, etc.) for the given structures.
        """
        ...


@runtime_checkable
class Trainer(Protocol):
    def train(self, dataset: Dataset, previous_potential: Path | None) -> Path:
        """
        Trains a potential model using the provided dataset.
        """
        ...


@runtime_checkable
class Validator(Protocol):
    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Validates the trained potential.
        """
        ...
