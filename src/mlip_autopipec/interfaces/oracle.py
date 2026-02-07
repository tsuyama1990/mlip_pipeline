from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseOracle(ABC):
    """
    Abstract base class for Oracle components (e.g., DFT, Mock).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        """
        Compute properties (energy, forces, stress) for a stream of structures.
        Should return a new stream of structures with properties filled.
        """
