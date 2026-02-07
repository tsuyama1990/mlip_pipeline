from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseGenerator(ABC):
    """
    Abstract Base Class for Structure Generators.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Generator component.

        Args:
            params: Dictionary of configuration parameters.
        """
        self.params = params or {}

    @abstractmethod
    def generate(self, count: int, workdir: Path) -> Iterator[Structure]:
        """
        Generate initial structures.

        Args:
            count: Number of structures to generate.
            workdir: Directory for artifacts.

        Returns:
            Iterator of generated Structure objects.
        """
