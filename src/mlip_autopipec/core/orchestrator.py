import logging
import shutil
from collections.abc import Iterator
from itertools import chain

from mlip_autopipec.constants import (
    DEFAULT_DATASET_FILENAME,
    DEFAULT_GENERATOR_COUNT,
    DEFAULT_POTENTIAL_EXTENSION,
)
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models import GlobalConfig, Potential, Structure
from mlip_autopipec.factory import create_dynamics, create_generator, create_oracle, create_trainer

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.workdir = config.workdir
        self.generator = create_generator(config.generator)
        self.oracle = create_oracle(config.oracle)
        self.trainer = create_trainer(config.trainer)
        self.dynamics = create_dynamics(config.dynamics)

        # Initialize dataset manager
        self.dataset_path = self.workdir / DEFAULT_DATASET_FILENAME
        self.dataset_manager = Dataset(self.dataset_path)

    def _load_dataset(self) -> Iterator[Structure]:
        """Loads structures from the dataset file lazily."""
        return iter(self.dataset_manager)

    def _safe_copy_potential(self, potential: Potential, cycle: int) -> None:
        """Safely copies potential file to workdir."""
        final_pot_path = self.workdir / f"potential_cycle_{cycle}{DEFAULT_POTENTIAL_EXTENSION}"
        if potential.path.exists():
             try:
                # Resolve paths to handle symlinks correctly if needed
                src = potential.path.resolve()
                dst = final_pot_path.resolve()
                # Security Check: Ensure destination is within workdir
                if not str(dst).startswith(str(self.workdir.resolve())):
                     logger.error(f"Security Warning: Attempted copy to {dst} outside workdir {self.workdir}")
                else:
                    shutil.copy(src, dst)
                    logger.info(f"Copied potential to {final_pot_path}")
             except Exception:
                logger.exception(f"Failed to copy potential to {final_pot_path}")

    def run(self) -> None:
        """
        Execute the active learning loop based on the configuration.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.dataset_manager._ensure_file()

        logger.info(f"Starting orchestration in {self.workdir}")

        # Initial Generation
        count = self.config.generator.get("count", DEFAULT_GENERATOR_COUNT)
        logger.info(f"Generating {count} initial structures...")

        # Stream structures directly to oracle
        initial_gen = self.generator.generate(count)

        try:
             first = next(initial_gen)
             initial_gen = chain([first], initial_gen)
        except StopIteration:
             logger.warning("No initial structures generated.")
             return

        # Initial Labeling
        logger.info("Labeling initial structures...")
        # Note: Oracle compute returns an iterator. We pass it directly to dataset append.
        # This keeps memory usage low as structures are processed one by one.
        labeled_iter = self.oracle.compute(initial_gen)
        self.dataset_manager.append(labeled_iter)
        logger.info(f"Dataset size: {self.dataset_manager.count()}")

        # Active Learning Loop
        for cycle in range(self.config.max_cycles):
            logger.info(f"--- Starting Cycle {cycle} ---")

            # Create cycle directory
            cycle_dir = self.workdir / f"cycle_{cycle}"
            cycle_dir.mkdir(parents=True, exist_ok=True)

            # Train
            logger.info("Training potential...")
            potential = self.trainer.train(self._load_dataset(), cycle_dir)
            logger.info(f"Potential trained at {potential.path}")

            self._safe_copy_potential(potential, cycle)

            # Dynamics / Exploration
            logger.info("Running dynamics exploration...")
            candidates_iter = self.dynamics.explore(potential)

            try:
                first_cand = next(candidates_iter)
                candidates_iter = chain([first_cand], candidates_iter)
                has_candidates = True
            except StopIteration:
                has_candidates = False

            if not has_candidates:
                logger.info("No new candidates found. Stopping.")
                break

            # Label new candidates
            logger.info("Labeling new candidates...")
            labeled_candidates_iter = self.oracle.compute(candidates_iter)
            self.dataset_manager.append(labeled_candidates_iter)
            logger.info(f"Dataset size: {self.dataset_manager.count()}")

            logger.info(f"Cycle {cycle} complete.")
