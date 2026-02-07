from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models import Potential, Structure


class BaseTrainer(ABC):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def train(self, structures: list[Structure], workdir: Path) -> Potential:
        """
        Train a potential model using the given dataset.
        """
