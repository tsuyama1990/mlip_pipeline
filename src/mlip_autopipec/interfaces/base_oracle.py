from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseOracle(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def compute(self, structures: list[Structure]) -> list[Structure]:
        """
        Compute properties (energy, forces, stress) for the given structures.
        Returns a list of structures with updated properties.
        """
