from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models import Dataset


class BaseOracle(ABC):
    def __init__(self, work_dir: Path | None = None) -> None:
        self.work_dir = work_dir

    @abstractmethod
    def label(self, dataset: Dataset) -> Dataset:
        """
        Computes energy, forces, and virial for the given structures.
        """
