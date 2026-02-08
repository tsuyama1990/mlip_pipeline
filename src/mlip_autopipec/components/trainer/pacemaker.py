import logging
from pathlib import Path

from mlip_autopipec.components.trainer.base import BaseTrainer
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.config import PacemakerTrainerConfig
from mlip_autopipec.domain_models.potential import Potential

logger = logging.getLogger(__name__)


class PacemakerTrainer(BaseTrainer):
    """
    Pacemaker implementation of the Trainer component.

    This component is responsible for training ACE potentials using the Pacemaker library.
    """

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
        """
        Train a potential using the provided dataset.

        Args:
            dataset: The dataset to train on.
            workdir: The directory to store training artifacts.
            previous_potential: Optional previous potential to start training from (fine-tuning).

        Returns:
            Potential: The trained potential object.

        Raises:
            NotImplementedError: Always, as this is a placeholder.
        """
        logger.error("Pacemaker Trainer is not yet implemented.")
        msg = "Pacemaker Trainer is not yet implemented."
        raise NotImplementedError(msg)
