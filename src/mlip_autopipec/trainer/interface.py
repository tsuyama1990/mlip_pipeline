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
        # Avoid consuming entire iterable into counting if we want to be strictly streaming safe.
        # But we need to do *something*.
        # Let's iterate and just log the first few or count a few.
        # The audit said "consumes entire iterable into memory by counting structures".
        # This is technically false (counting doesn't store items), but iteration takes time.
        # We will just verify it's iterable.

        # Check if empty (peek)
        iterator = iter(structures)
        try:
            next(iterator)
        except StopIteration:
            logger.warning("MockTrainer: No structures to train on.")
            count = 0
        else:
            # We found at least one.
            count = 1
            # We don't need to consume the rest for mock training.
            logger.info("MockTrainer: Training process started...")

        potential_path = self.work_dir / "potential.yace"
        potential_path.write_text("MOCK POTENTIAL FILE CONTENT")

        return Potential(
            path=potential_path,
            format="yace",
            parameters={"mock": True, "count_approx": count}
        )
