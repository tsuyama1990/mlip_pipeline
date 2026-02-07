from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models import Structure


class BaseOracle(ABC):
    """
    Abstract Base Class for Oracle components.
    """

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute properties (energy, forces, stress) for the given structures.

        Args:
            structures: An iterable of Structure objects.

        Returns:
            An iterator of Structure objects with computed properties.
        """
