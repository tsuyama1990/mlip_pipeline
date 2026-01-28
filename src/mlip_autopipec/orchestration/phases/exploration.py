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

                # Check for MD Engine (LammpsRunner)
                if self.config.inference_config:
                    from mlip_autopipec.inference.runner import LammpsRunner

                    potential_path = self.manager.state.latest_potential_path
                    if not potential_path or not potential_path.exists():
                        logger.error("No potential found for Active Learning.")
                        return

                    logger.info("Starting MD Exploration with LAMMPS...")
                    md_dir = self.manager.work_dir / f"md_cycle_{cycle}"
                    runner = LammpsRunner(self.config.inference_config, md_dir)

                    # Generate seeds using StructureBuilder
                    sys_config = SystemConfig(
                        target_system=self.config.target_system,
                        generator_config=self.config.generator_config
                    )
                    builder = getattr(self.manager, "builder", None) or StructureBuilder(sys_config)

                    # Generate a few seeds (e.g., 5)
                    # Use islice to limit the generator
                    seeds = list(itertools.islice(builder.build(), 5))

                    for i, atoms in enumerate(seeds):
                        uid = f"c{cycle}_md_{i}"
                        result = runner.run(atoms, potential_path, uid)

                        if result.halted:
                            logger.info(f"MD halted for {uid}. Capturing uncertain structures.")
                            for dump in result.uncertain_structures:
                                if dump.exists():
                                    self.manager.state.halted_structures.append(dump)

                    self.manager.save_state()

                elif self.config.surrogate_config:
                    logger.info("Running surrogate selection pipeline...")
                    surrogate = getattr(self.manager, "surrogate", None) or SurrogatePipeline(
                        self.db, self.config.surrogate_config
                    )
                    surrogate.run()
                else:
                    logger.warning("No inference config (MD) or surrogate config defined for Active Learning.")

        except Exception:
            logger.exception("Exploration phase failed")
