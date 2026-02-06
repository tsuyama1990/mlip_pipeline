from abc import ABC, abstractmethod

from mlip_autopipec.domain_models import Dataset


class BaseOracle(ABC):
    @abstractmethod
    def label(self, dataset: Dataset) -> Dataset:
        """
        Computes energy, forces, and virial for the given structures.

        Args:
            dataset (Dataset): The dataset containing structures to label.

        Returns:
            Dataset: A new dataset containing the labeled structures.
        """
