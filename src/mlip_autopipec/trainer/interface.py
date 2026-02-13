import logging
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path

from mlip_autopipec.domain_models.config import MOCK_POTENTIAL_CONTENT
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

    @abstractmethod
    def select_active_set(self, structures: Iterable[Structure], count: int) -> Iterator[Structure]:
        """
        Selects the most informative structures using D-Optimality (or random).

        Args:
            structures: Candidates to select from.
            count: Number of structures to select.

        Returns:
            An iterator of selected structures.
        """


class MockTrainer(BaseTrainer):
    """Mock implementation of Trainer."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(self, structures: Iterable[Structure]) -> Potential:
        logger.info("MockTrainer: Training process started (streaming mode)...")

        # Streaming with batching simulation
        batch_size = 100
        count = 0
        current_batch = []

        for s in structures:
            current_batch.append(s)
            if len(current_batch) >= batch_size:
                count += len(current_batch)
                # Flush batch to simulated disk/process
                current_batch = []

        # Flush remaining
        if current_batch:
            count += len(current_batch)

        logger.info(f"MockTrainer: Consumed {count} structures for training.")

        # Generate unique filename to avoid overwrites
        filename = f"potential_{uuid.uuid4().hex[:8]}.yace"
        potential_path = self.work_dir / filename

        # Validate file system permissions before writing
        if not self.work_dir.exists():
            # Should have been created in __init__, but check again
            msg = f"Work directory {self.work_dir} does not exist."
            raise FileNotFoundError(msg)

        # Check disk space (mock check, but good practice)
        total, used, free = shutil.disk_usage(self.work_dir)
        if free < 1024 * 1024:  # 1MB
            msg = f"Insufficient disk space in {self.work_dir}"
            raise OSError(msg)

        # Atomic write pattern
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=self.work_dir) as tmp:
                tmp.write(MOCK_POTENTIAL_CONTENT)
                tmp_path = Path(tmp.name)

            # Move to final destination (atomic on POSIX)
            shutil.move(str(tmp_path), str(potential_path))

        except OSError as e:
            msg = f"Failed to write potential file to {potential_path}: {e}"
            logger.exception(msg)
            raise

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
