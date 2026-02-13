import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
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

    def select_active_set(self, structures: Iterable[Structure], count: int) -> Iterator[Structure]:
        """
        Selects the most informative structures using D-Optimality (or random).
        Default implementation passes through the first N structures.

        Args:
            structures: Candidates to select from.
            count: Number of structures to select.

        Returns:
            An iterator of selected structures.
        """
        logger.warning("Using default select_active_set (pass-through). Subclasses should override.")

        for i, s in enumerate(structures):
            if i >= count:
                break
            yield s


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

    def select_active_set(self, structures: Iterable[Structure], count: int) -> Iterator[Structure]:
        logger.info(f"MockTrainer: Selecting {count} structures from candidates...")
        # Simple pass-through for mock
        for i, s in enumerate(structures):
            if i >= count:
                break
            # Mark provenance for tracking in tests
            s.provenance = f"{s.provenance}_selected"
            yield s
