from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.structure import Dataset


class BaseOracle(ABC):
    """
    Abstract Base Class for the Oracle (Ground Truth Generator).
    The Oracle takes a dataset of structures and computes their energy/forces/stress.
    """

    @abstractmethod
    def compute(self, dataset: Dataset) -> Dataset:
        """
        Computes the properties for the given dataset.

        Args:
            dataset: Input dataset with structures to compute.

        Returns:
            Dataset: New dataset with computed properties in structures' metadata/info.
        """
