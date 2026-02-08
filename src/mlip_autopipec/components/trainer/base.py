from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from mlip_autopipec.domain_models.config import TrainerConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.interfaces.base_component import BaseComponent

if TYPE_CHECKING:
    from mlip_autopipec.core.dataset import Dataset


class BaseTrainer(BaseComponent[TrainerConfig]):
    @property
    def name(self) -> str:
        return "trainer"

    @abstractmethod
    def train(
        self, dataset: "Dataset", workdir: Path, previous_potential: Potential | None = None
    ) -> Potential:
        """
        Train a potential.

        Args:
            dataset: The dataset to train on.
            workdir: Directory for training artifacts.
            previous_potential: Previous potential to resume from (optional).

        Returns:
            The trained Potential object.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
