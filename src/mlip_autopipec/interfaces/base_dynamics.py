from abc import ABC, abstractmethod
from collections.abc import Iterator

from mlip_autopipec.domain_models import Potential, Structure


class BaseDynamics(ABC):
    """
    Abstract base class for molecular dynamics engines.
    """

    @abstractmethod
    def run(self, potential: Potential) -> Iterator[Structure]:
        """
        Runs MD or other sampling dynamics using the given potential.

        Args:
            potential: The potential to use for dynamics.

        Returns:
            Iterator of sampled structures (e.g. uncertain configurations).
        """
