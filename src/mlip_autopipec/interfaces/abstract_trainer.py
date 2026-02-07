from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


class BaseTrainer(ABC):
    """
    Interface for training ML potentials.
    """

    @abstractmethod
    def train(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> Potential:
        """
        Trains a potential on the given structures.
        """
