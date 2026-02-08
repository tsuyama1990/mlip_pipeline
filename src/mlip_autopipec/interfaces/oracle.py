from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from mlip_autopipec.domain_models import Structure


class BaseOracle(ABC):
    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """Compute properties (energy, forces) for structures."""
