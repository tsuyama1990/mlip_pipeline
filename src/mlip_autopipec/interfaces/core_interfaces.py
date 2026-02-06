from pathlib import Path
from typing import Protocol

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.dataset import Dataset
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import ValidationResult


class Explorer(Protocol):
    """Protocol for structure generation."""

    def generate(self, config: GlobalConfig) -> list[StructureMetadata]:
        """
        Generate atomic structures based on configuration.

        Args:
            config: The global configuration.

        Returns:
            A list of generated structure metadata.

        Raises:
            RuntimeError: If generation fails.
        """
        ...


class Oracle(Protocol):
    """Protocol for property calculation."""

    def calculate(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """
        Calculate properties for the given structures.

        Args:
            structures: The structures to calculate properties for.

        Returns:
            The structures with calculated properties.

        Raises:
            RuntimeError: If calculation fails.
        """
        ...


class Trainer(Protocol):
    """Protocol for potential training."""

    def train(self, dataset: Dataset, previous_potential: Path | None) -> Path:
        """
        Train a potential on the given dataset.

        Args:
            dataset: The dataset to train on.
            previous_potential: Path to a previous potential to continue training from.

        Returns:
            The path to the trained potential artifact.

        Raises:
            RuntimeError: If training fails.
        """
        ...


class Validator(Protocol):
    """Protocol for potential validation."""

    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Validate a trained potential.

        Args:
            potential_path: The path to the potential to validate.

        Returns:
            The validation result.

        Raises:
            RuntimeError: If validation fails.
        """
        ...
