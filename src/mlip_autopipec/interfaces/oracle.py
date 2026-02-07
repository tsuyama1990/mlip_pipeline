from abc import ABC, abstractmethod
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseOracle(ABC):
    """
    Abstract base class for Oracles (calculators that provide energy and forces).
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def compute(self, structure: Structure) -> Structure:
        """
        Compute energy and forces for the given structure.
        Should return a new Structure with updated properties.
        """
