from pathlib import Path
from typing import Protocol, runtime_checkable

from ase import Atoms

from mlip_autopipec.domain_models import ValidationResult


@runtime_checkable
class Explorer(Protocol):
    """Interface for structure generation/exploration."""

    def explore(self, current_potential: Path | None) -> list[Atoms]:
        """
        Explore the configuration space to generate candidate structures.

        Args:
            current_potential: Path to the current potential file, or None if initial exploration.

        Returns:
            List[Atoms]: A list of generated atomic structures.
        """
        ...


@runtime_checkable
class Oracle(Protocol):
    """Interface for the Ground Truth provider (DFT)."""

    def compute(self, structures: list[Atoms]) -> list[Atoms]:
        """
        Compute properties (energy, forces, virial) for the given structures.

        Args:
            structures: List of atomic structures to compute.

        Returns:
            List[Atoms]: The structures with computed properties attached.
        """
        ...


@runtime_checkable
class Trainer(Protocol):
    """Interface for the Machine Learning Potential trainer."""

    def train(self, dataset: list[Atoms]) -> Path:
        """
        Train a potential on the given dataset.

        Args:
            dataset: List of labelled atomic structures.

        Returns:
            Path: The path to the trained potential file.
        """
        ...


@runtime_checkable
class Validator(Protocol):
    """Interface for potential validation."""

    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Validate the trained potential.

        Args:
            potential_path: Path to the potential file.

        Returns:
            ValidationResult: The result of the validation suite.
        """
        ...
