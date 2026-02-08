from abc import abstractmethod
from typing import Any

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseValidator(BaseComponent):
    @property
    def name(self) -> str:
        return "validator"

    @abstractmethod
    def validate(self, potential: Potential) -> dict[str, Any]:
        """Validate potential and return metrics."""
        ...
