from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models import Potential, ValidationResult


class BaseValidator(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def validate(self, potential: Potential) -> ValidationResult:
        """
        Validate the potential model.
        """
