from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential
from mlip_autopipec.domain_models.structure import Structure


class BaseDynamics(ABC):
    """
    Abstract Base Class for Dynamics (active learning exploration).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Dynamics component.

        Args:
            params: Dictionary of configuration parameters.
        """
        self.params = params or {}

    @abstractmethod
    def run(
        self,
        potential: Potential,
        initial_structures: Iterable[Structure],
        workdir: Path,
    ) -> ExplorationResult:
        """
        Run dynamics or exploration to find uncertain structures.

        Args:
            potential: The current potential to use for exploration.
            initial_structures: Starting structures for the exploration.
            workdir: Directory for artifacts.

        Returns:
            ExplorationResult containing new structures and convergence status.
        """
