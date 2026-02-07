from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models import Potential, Structure


class BaseTrainer(ABC):
    """
    Abstract base class for Potential Trainers.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        """
        Train a potential model using the provided dataset.
        """
