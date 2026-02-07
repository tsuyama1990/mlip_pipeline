from abc import ABC, abstractmethod
from collections.abc import Iterator

from mlip_autopipec.domain_models import Structure


class BaseGenerator(ABC):
    """
    Abstract Base Class for Structure Generation components.
    """

    @abstractmethod
    def generate(self) -> Iterator[Structure]:
        """
        Generate or load initial structures.

        Returns:
            An iterator of Structure objects.
        """
