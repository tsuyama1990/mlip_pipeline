from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseStructureGenerator(ABC):
    """
    Abstract base class for Structure Generators.
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def generate(self, base_structure: Structure, strategy: str) -> list[Structure]:
        """
        Generate new candidate structures based on a strategy.
        """
