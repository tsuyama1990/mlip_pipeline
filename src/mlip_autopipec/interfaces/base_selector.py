from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseSelector(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def select(self, candidates: list[Structure], n: int) -> list[Structure]:
        """
        Select a subset of candidate structures.
        """
