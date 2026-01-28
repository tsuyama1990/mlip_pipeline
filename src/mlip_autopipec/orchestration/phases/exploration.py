import itertools
import logging
from collections.abc import Iterable, Iterator
from typing import TypeVar

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.inference.runner import LammpsRunner
from mlip_autopipec.orchestration.phases.base import BasePhase

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

                # Check for necessary configs
                if not self.config.inference_config:
                    msg = "Inference config missing for Active Learning."
                    raise ValueError(msg)

                if not self.config.training_config:
                    msg = "Training config missing for Active Learning selection."
                    raise ValueError(msg)

                # Initialize Runner
                work_dir = self.manager.work_dir / f"exploration_gen_{cycle}"
                runner = LammpsRunner(self.config.inference_config, work_dir)

                potential_path = self.manager.state.latest_potential_path
                if not potential_path:
                    msg = "No potential available for exploration."
                    raise RuntimeError(msg)

                # Get starting structures
                initial_structures = list(self.db.get_atoms(selection="converged=True", limit=5))
                if not initial_structures:
                    logger.info("No converged structures found. Generating new ones.")
                    sys_config = SystemConfig(
                        target_system=self.config.target_system,
                        generator_config=self.config.generator_config
                    )
                    builder = StructureBuilder(sys_config)
                    try:
                        initial_structures = next(chunked(builder.build(), 5))
                    except StopIteration:
                        logger.warning("Builder produced no structures.")
                        initial_structures = []

                logger.info(f"Starting MD on {len(initial_structures)} structures.")

                for i, atoms in enumerate(initial_structures):
                    uid = f"md_{cycle}_{i}"
                    result = runner.run(atoms, potential_path, uid)

                    if result.halted:
                        logger.info(f"MD halted for {uid} at step {result.halt_step}. Queuing for selection...")
                        for dump_path in result.uncertain_structures:
                            self.manager.state.halted_structures.append(dump_path)
                        self.manager.save_state()

                    elif not result.succeeded:
                        logger.warning(f"MD failed for {uid}: {result.error_message}")

        except Exception:
            logger.exception("Exploration phase failed")
