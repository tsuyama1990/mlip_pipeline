from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from mlip_autopipec.domain_models import Potential, Structure


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        """Train a potential on the given dataset."""
