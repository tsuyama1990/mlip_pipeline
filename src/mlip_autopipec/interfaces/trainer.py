from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlip_autopipec.domain_models import Structure


class BaseTrainer(ABC):
    """
    Abstract base class for Trainers (fitting MLIPs).
    """
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    @abstractmethod
    def train(self, structures: Iterable[Structure], params: dict[str, Any], workdir: str | Path) -> Path:
        """
        Train a potential on the given structures.
        Returns the path to the trained potential file (e.g. .yace).

        Note: Implementations should iterate over `structures` and avoid loading everything into memory if possible,
        or handle serialization to disk incrementally.
        """
