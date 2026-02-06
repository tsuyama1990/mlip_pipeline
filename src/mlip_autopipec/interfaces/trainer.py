from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models import Dataset


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, dataset: Dataset, validation_set: Dataset | None = None) -> Path:
        pass
