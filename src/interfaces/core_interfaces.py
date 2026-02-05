from pathlib import Path
from typing import Protocol

from config import GlobalConfig
from domain_models import Dataset, StructureMetadata
from domain_models.dataset import ValidationResult


class Explorer(Protocol):
    def generate(self, config: GlobalConfig) -> list[StructureMetadata]: ...


class Oracle(Protocol):
    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]: ...


class Trainer(Protocol):
    def train(self, dataset: Dataset, previous_potential: Path | None) -> Path: ...


class Validator(Protocol):
    def validate(self, potential_path: Path) -> ValidationResult: ...
