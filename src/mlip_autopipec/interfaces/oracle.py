from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseOracle(ABC):
    """
    Abstract Base Class for Oracles (calculators of energy/forces/stress).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Oracle component.

        Args:
            params: Dictionary of configuration parameters.
        """
        self.params = params or {}

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute properties for a batch of structures.

        Args:
            structures: Iterable of Structure objects.

        Returns:
            Iterator of Structure objects with computed properties.
        """
