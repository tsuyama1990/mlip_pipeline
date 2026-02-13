import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path

from mlip_autopipec.constants import MOCK_POTENTIAL_CONTENT
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
        logger.warning(
            "Using default select_active_set (pass-through). Subclasses should override."
        )

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
        logger.info("MockTrainer: Training process started (streaming mode)...")

        # In a real scenario, we would stream this to a file or tool.
        # For mock, we simply acknowledge the iterable exists.
        # We MUST consume the iterator to drive the pipeline (Oracle -> Trainer).
        count = 0
        for _ in structures:
            count += 1
        logger.info(f"MockTrainer: Consumed {count} structures for training.")

        # Generate unique filename to avoid overwrites
        filename = f"potential_{uuid.uuid4().hex[:8]}.yace"
        potential_path = self.work_dir / filename
        potential_path.write_text(MOCK_POTENTIAL_CONTENT)

        return Potential(path=potential_path, format="yace", parameters={"mock": True})

    def select_active_set(self, structures: Iterable[Structure], count: int) -> Iterator[Structure]:
        logger.info(f"MockTrainer: Selecting {count} structures from candidates...")
        # Simple pass-through for mock
        for i, s in enumerate(structures):
            if i >= count:
                break
            # Mark provenance for tracking in tests
            s.provenance = f"{s.provenance}_selected"
            yield s
