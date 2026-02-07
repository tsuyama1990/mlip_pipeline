from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from mlip_autopipec.domain_models import Potential, Structure


class BaseTrainer(ABC):
    """
    Abstract Base Class for Trainer components.
    """

    @abstractmethod
    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        """
        Train a potential model using the given dataset.

        Args:
            dataset: An iterable of Structure objects.
            workdir: Directory to store training artifacts.

        Returns:
            A Potential object representing the trained model.
        """
