from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


class BaseTrainer(ABC):
    """
    Abstract Base Class for Trainers (MLIP fitting engines).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Trainer component.

        Args:
            params: Dictionary of configuration parameters.
        """
        self.params = params or {}

    @abstractmethod
    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        """
        Train a potential on the given dataset.

        Args:
            dataset: Iterable of Structure objects.
            workdir: Directory to store training artifacts.

        Returns:
            Potential object pointing to the trained model.
        """
