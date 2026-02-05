from pathlib import Path
from typing import Protocol

from src.config.config_model import GlobalConfig
from src.domain_models import Dataset, StructureMetadata, ValidationResult


class Explorer(Protocol):
    def generate(self, config: GlobalConfig) -> list[StructureMetadata]:
        """Generate new atomic structures."""
        ...


class Oracle(Protocol):
    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Calculate properties (energy, forces) for structures."""
        ...


class Trainer(Protocol):
    def train(self, dataset: Dataset, previous_potential: Path | None) -> Path:
        """Train a potential on the dataset."""
        ...


class Validator(Protocol):
    def validate(self, potential_path: Path) -> ValidationResult:
        """Validate the potential."""
        ...
