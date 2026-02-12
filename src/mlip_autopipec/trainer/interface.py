import logging
from abc import ABC, abstractmethod
from pathlib import Path

from mlip_autopipec.domain_models.datastructures import Dataset, Potential

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract Base Class for Potential Trainer."""

    @abstractmethod
    def train(self, dataset: Dataset) -> Potential:
        """
        Trains a potential on the given dataset.

        Args:
            dataset: The dataset to train on.

        Returns:
            A Potential object representing the trained model.
        """


class MockTrainer(BaseTrainer):
    """Mock implementation of Trainer."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(self, dataset: Dataset) -> Potential:
        logger.info(f"MockTrainer: Training on {len(dataset)} structures...")

        potential_path = self.work_dir / "potential.yace"
        potential_path.write_text("MOCK POTENTIAL FILE CONTENT")

        return Potential(
            path=potential_path,
            format="yace",
            parameters={"mock": True}
        )
