from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models import ExplorationResult, Potential, Structure


class BaseDynamics(ABC):
    """
    Abstract base class for Molecular Dynamics / Exploration.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def run(
        self, potential: Potential, start_structure: Structure, workdir: Path
    ) -> ExplorationResult:
        """
        Run dynamics exploration starting from a structure using a potential.
        """
