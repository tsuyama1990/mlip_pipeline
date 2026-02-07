from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models import Potential, Structure, ValidationResult


class BaseValidator(ABC):
    """
    Abstract base class for potential validation.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def validate(
        self, potential: Potential, test_set: Iterable[Structure], workdir: Path
    ) -> ValidationResult:
        """
        Validate a potential model against a test set.
        """
