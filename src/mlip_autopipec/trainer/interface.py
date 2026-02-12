import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from mlip_autopipec.domain_models.datastructures import Potential, Structure

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract Base Class for Potential Trainer."""

    @abstractmethod
    def train(self, structures: Iterable[Structure]) -> Potential:
        """
        Trains a potential on the given structures.

        Args:
            structures: An iterable of labeled structures.

        Returns:
            A Potential object representing the trained model.
        """


class MockTrainer(BaseTrainer):
    """Mock implementation of Trainer."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(self, structures: Iterable[Structure]) -> Potential:
        # In a streaming scenario, we might write to disk chunk by chunk.
        # For mock, we just count or consume.
        # We need to consume the iterable to "train".
        count = 0
        for _ in structures:
            count += 1

        logger.info(f"MockTrainer: Training on {count} structures...")

        potential_path = self.work_dir / "potential.yace"
        potential_path.write_text("MOCK POTENTIAL FILE CONTENT")

        return Potential(
            path=potential_path,
            format="yace",
            parameters={"mock": True, "count": count}
        )
