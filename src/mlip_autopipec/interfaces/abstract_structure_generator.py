from abc import ABC, abstractmethod

from mlip_autopipec.domain_models.structure import Structure


class BaseStructureGenerator(ABC):
    """
    Interface for generating candidate structures.
    """

    @abstractmethod
    def get_candidates(self) -> list[Structure]:
        """
        Generates a list of candidate structures for labeling.
        """
