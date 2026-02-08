from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models import Potential


class BaseValidator(ABC):
    """
    Abstract base class for validation engines.
    """

    @abstractmethod
    def validate(self, potential: Potential) -> dict[str, Any]:
        """
        Validates the potential (e.g., computes metrics on a test set).

        Args:
            potential: The potential to validate.

        Returns:
            Dictionary of validation metrics.
        """
