from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from mlip_autopipec.domain_models import Potential, Structure, ValidationResult


class BaseValidator(ABC):
    """
    Abstract Base Class for Validation components.
    """

    @abstractmethod
    def validate(
        self, potential: Potential, test_set: Iterable[Structure], workdir: Path
    ) -> ValidationResult:
        """
        Validate a potential model against a test set.

        Args:
            potential: The potential model to validate.
            test_set: An iterable of Structure objects for testing.
            workdir: Directory for validation artifacts.

        Returns:
            A ValidationResult object.
        """
