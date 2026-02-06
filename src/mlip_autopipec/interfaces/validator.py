from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models import ValidationResult


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, potential_path: Path) -> ValidationResult:
        pass
