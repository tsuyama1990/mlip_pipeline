from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult


class BaseValidator(ABC):
    """
    Abstract Base Class for Validation components.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Validator component.

        Args:
            params: Dictionary of configuration parameters.
        """
        self.params = params or {}

    @abstractmethod
    def validate(
        self,
        potential: Potential,
        test_set: Iterable[Structure],
        workdir: Path,
    ) -> ValidationResult:
        """
        Validate the potential against a test set.

        Args:
            potential: The potential to validate.
            test_set: The structures to test against.
            workdir: Directory for artifacts.

        Returns:
            ValidationResult containing metrics and pass/fail status.
        """
