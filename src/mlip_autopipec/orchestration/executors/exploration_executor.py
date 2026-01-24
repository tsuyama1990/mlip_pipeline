import itertools
import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.orchestration.executors.base_executor import BaseExecutor
from mlip_autopipec.orchestration.interfaces import BuilderProtocol, SurrogateProtocol
from mlip_autopipec.surrogate.pipeline import SurrogatePipeline

logger = logging.getLogger(__name__)

T = TypeVar("T")

def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

class ExplorationExecutor(BaseExecutor):
    """Executes Phase A: Exploration."""

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self._builder: BuilderProtocol | None = manager.builder
        self._surrogate: SurrogateProtocol | None = manager.surrogate

    def execute(self) -> bool:
        """Execute Phase A: Exploration."""
        logger.info("Phase A: Exploration")
        try:
            if not self._builder:
                self._builder = StructureBuilder(self.config)

            batch_size = 100
            total_generated = 0

            # Chunked processing
            for candidate_batch in chunked(self._builder.build(), batch_size):
                if self.config.surrogate_config:
                    if not self._surrogate:
                        self._surrogate = SurrogatePipeline(self.config.surrogate_config)

                    selected, _ = self._surrogate.run(candidate_batch)
                    logger.info(f"Batch: Generated {len(candidate_batch)}, Selected {len(selected)}")
                else:
                    selected = candidate_batch

                for atoms in selected:
                    self.db.save_candidate(
                        atoms,
                        {"status": "pending", "generation": self.manager.state.current_generation},
                    )
                total_generated += len(selected)

            logger.info(f"Exploration complete. Total candidates saved: {total_generated}")
            return total_generated > 0

        except Exception:
            logger.exception("Exploration phase failed")
            return False
