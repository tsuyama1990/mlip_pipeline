from abc import ABC, abstractmethod
from collections.abc import Iterator

from mlip_autopipec.domain_models import Potential, Structure


class BaseDynamics(ABC):
    @abstractmethod
    def explore(self, potential: Potential) -> Iterator[Structure]:
        """Run dynamics/exploration using the potential and return new structures."""
