from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.validation import ValidationResult


class BaseValidator(ABC):
    """
    Interface for validating ML potentials.
    """

    @abstractmethod
    def validate(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ValidationResult:
        """
        Validates the potential against a test set or other metrics.
        """
