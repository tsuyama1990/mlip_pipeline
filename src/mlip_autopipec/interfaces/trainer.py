from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from mlip_autopipec.domain_models import Dataset

class BaseTrainer(ABC):
    @abstractmethod
    def train(self, dataset: Dataset, validation_set: Optional[Dataset] = None) -> Path:
        pass
