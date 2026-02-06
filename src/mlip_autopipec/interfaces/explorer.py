from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models import Dataset


class BaseExplorer(ABC):
    @abstractmethod
    def explore(self, current_potential_path: Path, dataset: Dataset) -> Dataset:
        """
        Generates new candidate structures using the current potential and dataset.

        Args:
            current_potential_path (Path): Path to the current potential file.
            dataset (Dataset): The current accumulated dataset.

        Returns:
            Dataset: A dataset containing the newly generated candidate structures.
        """
