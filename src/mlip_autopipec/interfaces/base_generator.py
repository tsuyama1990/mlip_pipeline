from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseStructureGenerator(ABC):
    """
    Abstract base class for Structure Generator (perturbations, mutations, etc.).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the generator with parameters.
        """
        self.params = params or {}

    @abstractmethod
    def generate(self, source: Structure) -> Iterator[Structure]:
        """
        Generate a stream of new structures based on a source structure.

        Args:
            source: The input Structure to be modified or perturbed.

        Returns:
            An iterator yielding new Structure objects.
        """
