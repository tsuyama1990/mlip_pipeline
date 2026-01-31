from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.validation import ValidationResult


class BaseValidator(ABC):
    """Abstract base class for all physics validators."""

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Run the validation and return the result."""
        pass
