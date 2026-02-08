from abc import ABC, abstractmethod
from collections.abc import Iterator

from mlip_autopipec.domain_models import Structure


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, count: int) -> Iterator[Structure]:
        """Generate candidate structures."""
