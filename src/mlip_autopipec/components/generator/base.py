from abc import abstractmethod
from typing import Any

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseGenerator(BaseComponent):
    @property
    def name(self) -> str:
        return "generator"

    @abstractmethod
    def generate(self, n_structures: int, config: dict[str, Any] | None = None) -> list[Structure]:
        """Generate structures."""
        ...
