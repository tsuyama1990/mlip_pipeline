from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential


class BaseDynamics(ABC):
    """
    Interface for molecular dynamics or other exploration methods.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Dynamics engine.

        Args:
            params: Dictionary of parameters for the Dynamics implementation.
        """
        self.params = params or {}

    @abstractmethod
    def run_exploration(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ExplorationResult:
        """
        Runs exploration using the given potential.

        Args:
            potential: The potential to explore.
            workdir: Directory to store exploration artifacts.

        Returns:
            An ExplorationResult object containing the outcome of the exploration.
        """
