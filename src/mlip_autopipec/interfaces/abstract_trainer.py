from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


class BaseTrainer(ABC):
    """
    Interface for training ML potentials.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Trainer.

        Args:
            params: Dictionary of parameters for the Trainer implementation.
        """
        self.params = params or {}

    @abstractmethod
    def train(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> Potential:
        """
        Trains a potential on the given structures.

        Args:
            structures: List of structures to train on.
            workdir: Directory to store training artifacts.

        Returns:
            A Potential object representing the trained model.
        """
