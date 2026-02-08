from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models import Structure


class BaseOracle(ABC):
    """
    Abstract base class for oracles (e.g. DFT calculators).
    """

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Computes properties (energy, forces, stress) for the given structures.

        Args:
            structures: Iterable of structures to compute.

        Returns:
            Iterator of computed structures (with updated labels).
        """
