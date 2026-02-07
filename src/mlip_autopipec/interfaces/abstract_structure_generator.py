from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseStructureGenerator(ABC):
    """
    Interface for generating candidate structures.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Structure Generator.

        Args:
            params: Dictionary of parameters for the generator implementation.
        """
        self.params = params or {}

    @abstractmethod
    def get_candidates(self) -> list[Structure]:
        """
        Generates a list of candidate structures for labeling.

        Returns:
            A list of new candidate structures.
        """
