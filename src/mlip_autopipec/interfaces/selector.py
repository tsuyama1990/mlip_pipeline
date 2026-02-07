from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseSelector(ABC):
    """
    Abstract Base Class for Active Learning Selectors.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Selector component.

        Args:
            params: Dictionary of configuration parameters.
        """
        self.params = params or {}

    @abstractmethod
    def select(
        self,
        candidates: Iterable[Structure],
        count: int,
    ) -> Iterator[Structure]:
        """
        Select the most informative structures from candidates.

        Args:
            candidates: Iterable of candidate structures.
            count: Maximum number of structures to select.

        Returns:
            Iterator of selected structures.
        """
