"""Base validator interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.domain_models.validator import ValidationResult


class BaseValidator(ABC):
    """Base validator interface."""

    @abstractmethod
    def validate(
        self,
        potential_path: Path,
        structure: Atoms,
        output_dir: Path,
        **kwargs: Any,
    ) -> ValidationResult:
        """Run validation."""
        ...
