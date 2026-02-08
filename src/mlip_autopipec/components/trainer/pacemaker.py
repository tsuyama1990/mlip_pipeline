from pathlib import Path

from mlip_autopipec.components.trainer.base import BaseTrainer
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.config import PacemakerTrainerConfig
from mlip_autopipec.domain_models.potential import Potential


class PacemakerTrainer(BaseTrainer):
    def __init__(self, config: PacemakerTrainerConfig) -> None:
        super().__init__(config)
        self.config: PacemakerTrainerConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def train(
        self,
        dataset: Dataset,
        workdir: Path,
        previous_potential: Potential | None = None,
    ) -> Potential:
        msg = "Pacemaker Trainer is not yet implemented."
        raise NotImplementedError(msg)
