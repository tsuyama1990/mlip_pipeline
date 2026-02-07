from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models import ExplorationResult, Structure


class BaseDynamics(ABC):
    """
    Abstract base class for Dynamics (exploration engines).
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def run(self, potential: str | Path, structure: Structure) -> ExplorationResult:
        """
        Run dynamics or exploration using the given potential and initial structure.
        """
