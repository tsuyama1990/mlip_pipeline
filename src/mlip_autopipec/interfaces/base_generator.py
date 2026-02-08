from abc import ABC, abstractmethod
from collections.abc import Iterator

from mlip_autopipec.domain_models import Potential, Structure


class BaseGenerator(ABC):
    """
    Abstract base class for structure generators.
    """

    @abstractmethod
    def generate(self, potential: Potential | None = None) -> Iterator[Structure]:
        """
        Generates candidate structures.

        Args:
            potential: Optional potential to guide generation (e.g. for active learning).

        Returns:
            Iterator of generated Structure objects.
        """
