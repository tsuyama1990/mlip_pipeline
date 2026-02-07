from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models import Dataset


class BaseTrainer(ABC):
    """
    Abstract base class for Trainers (fitting MLIPs).
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def train(self, dataset: Dataset, params: dict[str, Any], workdir: str | Path) -> Path:
        """
        Train a potential on the given dataset.
        Returns the path to the trained potential file (e.g. .yace).
        """
