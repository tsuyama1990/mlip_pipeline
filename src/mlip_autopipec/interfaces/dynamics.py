from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from mlip_autopipec.domain_models import ExplorationResult, Potential, Structure


class BaseDynamics(ABC):
    """
    Abstract Base Class for Dynamics/Exploration components.
    """

    @abstractmethod
    def run(
        self,
        potential: Potential,
        initial_structures: Iterable[Structure],
        workdir: Path,
    ) -> ExplorationResult:
        """
        Run dynamics or exploration using the given potential.

        Args:
            potential: The potential model to use for exploration.
            initial_structures: Starting configurations for exploration.
            workdir: Directory for exploration artifacts.

        Returns:
            An ExplorationResult object containing status and new structures.
        """
