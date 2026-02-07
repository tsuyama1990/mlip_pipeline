from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models.structure import Structure


class BaseOracle(ABC):
    """
    Interface for the Oracle (e.g., DFT).
    """

    @abstractmethod
    def compute(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> list[Structure]:
        """
        Takes a list of structures and returns them with computed
        energy, forces, and stress.
        """
