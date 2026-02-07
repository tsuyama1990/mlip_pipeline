from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.validation import ValidationResult


class BaseValidator(ABC):
    """
    Interface for validating ML potentials.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Validator.

        Args:
            params: Dictionary of parameters for the validation implementation.
        """
        self.params = params or {}

    @abstractmethod
    def validate(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ValidationResult:
        """
        Validates the potential against a test set or other metrics.

        Args:
            potential: The potential to validate.
            workdir: Directory to store validation artifacts.

        Returns:
            A ValidationResult object indicating pass/fail and metrics.
        """
