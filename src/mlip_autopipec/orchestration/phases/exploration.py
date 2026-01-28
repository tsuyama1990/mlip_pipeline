import itertools
import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.orchestration.phases.base import BasePhase
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


class ExplorationPhase(BasePhase):
    def execute(self) -> None:
        """Execute Phase A: Exploration."""
        logger.info("Phase A: Exploration")
        try:
            builder = self.manager.builder or StructureBuilder(self.config)

            # TODO: Move batch_size to config
            batch_size = 100
            total_generated = 0

            # Chunked processing to avoid OOM
            for candidate_batch in chunked(builder.build(), batch_size):
                for atoms in candidate_batch:
                    self.db.save_candidate(
                        atoms,
                        {"status": "pending", "generation": self.manager.state.cycle_index},
                    )
                total_generated += len(candidate_batch)

            if self.config.surrogate_config:
                logger.info("Running surrogate selection pipeline...")
                surrogate = self.manager.surrogate or SurrogatePipeline(
                    self.db, self.config.surrogate_config
                )
                surrogate.run()

            logger.info(f"Exploration complete. Total candidates generated: {total_generated}")

        except Exception:
            logger.exception("Exploration phase failed")
