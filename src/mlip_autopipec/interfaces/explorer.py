from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential


class BaseExplorer(ABC):
    """
    Abstract Base Class for the Explorer (Structure Generator/Dynamics Engine).
    The Explorer uses a potential to run simulations (MD/kMC) and generates new structures
    in regions of high uncertainty.
    """

    @abstractmethod
    def explore(self, potential: Potential) -> ExplorationResult:
        """
        Runs an exploration simulation using the given potential.

        Args:
            potential: The potential to use for exploration.

        Returns:
            ExplorationResult: Result containing halt status, dump file path, and other metrics.
        """
