from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Dataset


class BaseTrainer(ABC):
    """
    Abstract Base Class for the Trainer (MLIP Learner).
    The Trainer fits an interatomic potential to a labeled dataset.
    """

    @abstractmethod
    def train(self, dataset: Dataset, previous_potential: Potential | None = None) -> Potential:
        """
        Trains a new potential using the dataset.

        Args:
            dataset: Labeled dataset for training.
            previous_potential: Optional previous potential to use as starting point/baseline.

        Returns:
            Potential: The newly trained potential artifact.
        """
