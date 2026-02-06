from abc import ABC, abstractmethod
from pathlib import Path
from mlip_autopipec.domain_models import Dataset

class BaseTrainer(ABC):
    @abstractmethod
    def train(self, train_dataset: Dataset, validation_dataset: Dataset) -> Path:
        """
        Trains a potential using the provided datasets.
        Returns the path to the trained potential file.
        """
        pass
