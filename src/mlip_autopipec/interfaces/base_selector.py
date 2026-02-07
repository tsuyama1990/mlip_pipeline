from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseSelector(ABC):
    """
    Abstract base class for Active Learning Selector.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the selector with parameters.
        """
        self.params = params or {}

    @abstractmethod
    def select(self, candidates: Iterable[Structure], n: int) -> Iterator[Structure]:
        """
        Select a subset of structures from a candidate pool for labeling.

        Args:
            candidates: Iterable of Structure objects.
            n: Number of structures to select.

        Returns:
            An iterator yielding the selected Structure objects.
        """
