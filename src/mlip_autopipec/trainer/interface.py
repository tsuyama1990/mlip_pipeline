import logging
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure

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

        # True Streaming: Count structures without accumulating them in memory
        count = 0
        for _ in structures:
            # In a real scenario, here we would stream-write 's' to a dataset file
            # e.g., dataset_writer.write(s)
            count += 1

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

        # Atomic write pattern with unique temp directory
        tmp_dir = Path(tempfile.mkdtemp(dir=self.work_dir))
        try:
            tmp_path = tmp_dir / "temp_potential.yace"

            # Use content from config if available, else default
            # MockTrainer doesn't have full TrainerConfig usually, but if it did:
            content = "MOCK_POTENTIAL_CONTENT"
            if hasattr(self, 'config') and hasattr(self.config, 'mock_potential_content'):
                 content = self.config.mock_potential_content

            tmp_path.write_text(content)

            # Atomic move (rename)
            # rename is atomic on POSIX if on same filesystem (guaranteed by tmp_dir inside work_dir)
            tmp_path.rename(potential_path)

        except OSError as e:
            msg = f"Failed to write potential file to {potential_path}: {e}"
            logger.exception(msg)
            raise
        finally:
            # Cleanup temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
