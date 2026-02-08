from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from mlip_autopipec.domain_models import Potential, Structure


class BaseTrainer(ABC):
    """
    Abstract base class for potential trainers.
    """

    @abstractmethod
    def train(
        self,
        dataset: Iterable[Structure],
        initial_potential: Potential | None = None,
        workdir: Path | None = None,
    ) -> Potential:
        """
        Trains a potential using the given dataset.

        Args:
            dataset: Iterable of labeled structures.
            initial_potential: Optional starting potential for transfer learning.
            workdir: Directory to store training artifacts.

        Returns:
            The trained Potential object.
        """
