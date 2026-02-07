from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseStructureGenerator(ABC):
    """
    Abstract base class for structure generation.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def generate(self, n: int = 1) -> Iterator[Structure]:
        """
        Generate `n` initial structures.
        """
