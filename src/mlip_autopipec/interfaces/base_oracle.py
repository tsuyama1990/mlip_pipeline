from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseOracle(ABC):
    """
    Abstract base class for Oracle (energy/forces calculator).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Oracle with parameters.
        """
        self.params = params or {}

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute properties (energy, forces, stress) for a stream of structures.

        Args:
            structures: An iterable of Structure objects.

        Returns:
            An iterator yielding Structure objects with updated properties.
        """
