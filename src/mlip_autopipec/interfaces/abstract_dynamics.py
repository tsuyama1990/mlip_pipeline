from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential


class BaseDynamics(ABC):
    """
    Interface for molecular dynamics or other exploration methods.
    """

    @abstractmethod
    def run_exploration(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ExplorationResult:
        """
        Runs exploration using the given potential.
        """
