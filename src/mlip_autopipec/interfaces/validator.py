from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.validation import ValidationResult


class BaseValidator(ABC):
    """
    Abstract base class for Validators (Quality Assurance).
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Validate the trained potential.
        """
