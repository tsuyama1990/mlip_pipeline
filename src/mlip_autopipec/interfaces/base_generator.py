from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseStructureGenerator(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def generate(self, structure: Structure) -> list[Structure]:
        """
        Generate new candidate structures from the source structure.
        """
