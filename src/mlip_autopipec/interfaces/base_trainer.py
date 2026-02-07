from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


class BaseTrainer(ABC):
    """
    Abstract base class for Potential Trainer.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize the Trainer with parameters.
        """
        self.params = params or {}

    @abstractmethod
    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        """
        Train a potential model using the provided dataset.

        Args:
            dataset: An iterable of labeled Structure objects.
            workdir: Directory to store training artifacts and the final model.

        Returns:
            A Potential object representing the trained model.
        """
