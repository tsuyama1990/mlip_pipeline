from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.structure import Structure


class BaseOracle(ABC):
    """
    Interface for the Oracle (e.g., DFT).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Oracle.

        Args:
            params: Dictionary of parameters for the Oracle implementation.
        """
        self.params = params or {}

    @abstractmethod
    def compute(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> list[Structure]:
        """
        Takes a list of structures and returns them with computed
        energy, forces, and stress.

        Args:
            structures: List of structures to compute properties for.
            workdir: Directory to store calculation artifacts.

        Returns:
            The input list of structures with computed properties added.
        """
