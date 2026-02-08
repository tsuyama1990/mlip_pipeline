from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.interfaces.base_component import BaseComponent

if TYPE_CHECKING:
    from mlip_autopipec.core.dataset import Dataset


class BaseTrainer(BaseComponent):
    @property
    def name(self) -> str:
        return "trainer"

    @abstractmethod
    def train(
        self, dataset: "Dataset", workdir: Path, previous_potential: Potential | None = None
    ) -> Potential:
        """Train a potential."""
        ...
