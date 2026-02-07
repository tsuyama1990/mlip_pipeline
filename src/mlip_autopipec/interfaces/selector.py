from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseSelector(ABC):
    """
    Abstract base class for active learning selection strategies.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def select(self, candidates: Iterable[Structure], n: int) -> Iterator[Structure]:
        """
        Select the most informative structures from a candidate pool.
        """
