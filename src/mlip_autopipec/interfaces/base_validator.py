from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult


class BaseValidator(ABC):
    """
    Abstract base class for validating a potential model.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the validator with parameters.
        """
        self.params = params or {}

    @abstractmethod
    def validate(self, potential: Potential, dataset: Iterable[Structure]) -> ValidationResult:
        """
        Validate a potential model against a validation dataset.

        Args:
            potential: The trained potential model.
            dataset: Iterable of Structure objects with ground truth labels.

        Returns:
            ValidationResult containing success status, metrics, and report details.
        """
