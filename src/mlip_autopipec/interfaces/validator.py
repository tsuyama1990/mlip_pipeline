from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models import ValidationResult


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Validates the trained potential.

        Args:
            potential_path (Path): The path to the potential to validate.

        Returns:
            ValidationResult: The results of the validation.
        """
