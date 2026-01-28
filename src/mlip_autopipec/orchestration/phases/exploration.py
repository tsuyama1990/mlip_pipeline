import itertools
import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

from mlip_autopipec.config.models import SystemConfig
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
            cycle = self.manager.state.cycle_index

            if cycle == 0:
                # Cold Start: Structure Generation
                logger.info("Cycle 0: Running Structure Generator (Cold Start)")

                sys_config = SystemConfig(
                    target_system=self.config.target_system,
                    generator_config=self.config.generator_config
                )
                # Allow injection or default
                builder = getattr(self.manager, "builder", None) or StructureBuilder(sys_config)

                # TODO: Move batch_size to config
                batch_size = 100
                total_generated = 0

                # Chunked processing to avoid OOM
                for candidate_batch in chunked(builder.build(), batch_size):
                    for atoms in candidate_batch:
                        # Fix: use add_structure instead of non-existent save_candidate
                        self.db.add_structure(
                            atoms,
                            {"status": "pending", "generation": cycle},
                        )
                    total_generated += len(candidate_batch)

                logger.info(f"Cold Start complete. Total candidates generated: {total_generated}")

            else:
                # Active Learning Exploration
                logger.info(f"Cycle {cycle}: Running Active Learning Exploration")

                if self.config.surrogate_config:
                    logger.info("Running surrogate selection pipeline...")
                    surrogate = getattr(self.manager, "surrogate", None) or SurrogatePipeline(
                        self.db, self.config.surrogate_config
                    )
                    surrogate.run()
                else:
                    logger.warning("No surrogate config or MD engine defined for Active Learning.")

        except Exception:
            logger.exception("Exploration phase failed")
