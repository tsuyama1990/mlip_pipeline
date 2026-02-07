from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseSelector(ABC):
    """
    Abstract base class for Selectors (Active Learning Selection).
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def select(self, candidates: list[Structure], n: int, existing_data: list[Structure] | None = None) -> list[Structure]:
        """
        Select the most informative structures from candidates.
        """
