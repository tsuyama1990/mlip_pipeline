from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential
from mlip_autopipec.domain_models.structure import Structure


class BaseDynamics(ABC):
    """
    Abstract base class for Molecular Dynamics or similar exploration.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Dynamics engine with parameters.
        """
        self.params = params or {}

    @abstractmethod
    def run(
        self, potential: Potential, start_structure: Structure, workdir: Path
    ) -> ExplorationResult:
        """
        Run dynamics simulation starting from a structure using a potential.

        Args:
            potential: The potential model to use for forces/energy.
            start_structure: The initial atomic configuration.
            workdir: Directory for any temporary files or trajectory output.

        Returns:
            ExplorationResult containing status, final structure (if halted), and trajectory.
        """
