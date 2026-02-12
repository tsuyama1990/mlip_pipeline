import logging
from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.datastructures import Potential, ValidationResult

logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """Abstract Base Class for Potential Validator."""

    @abstractmethod
    def validate(self, potential: Potential) -> ValidationResult:
        """
        Validates the potential against physics constraints.

        Args:
            potential: The potential to validate.

        Returns:
            A ValidationResult object.
        """


class MockValidator(BaseValidator):
    """Mock implementation of Validator."""

    def validate(self, potential: Potential) -> ValidationResult:
        logger.info(f"MockValidator: Validating potential {potential.path}...")

        return ValidationResult(
            passed=True,
            metrics={
                "elastic_error": 0.05,
                "phonon_stability": 1.0,
                "eos_error": 0.01
            },
            report_path=None
        )
