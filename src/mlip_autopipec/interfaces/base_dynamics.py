from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models import ExplorationResult, Potential, Structure


class BaseDynamics(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def run(self, potential: Potential, structure: Structure) -> ExplorationResult:
        """
        Run molecular dynamics simulation using the potential.
        """
