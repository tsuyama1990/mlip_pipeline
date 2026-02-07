from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.validation import ValidationResult


class BaseValidator(ABC):
    """
    Abstract Base Class for the Validator (Quality Assurance).
    The Validator runs physics-based checks (phonons, elasticity, etc.) on the potential.
    """

    @abstractmethod
    def validate(self, potential: Potential) -> ValidationResult:
        """
        Validates the given potential against a set of checks.

        Args:
            potential: The potential to validate.

        Returns:
            ValidationResult: Result of the validation checks.
        """
