import logging
import shutil

from mlip_autopipec.domain_models import GlobalConfig, Potential, Structure
from mlip_autopipec.factory import create_dynamics, create_generator, create_oracle, create_trainer

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.workdir = config.workdir
        self.generator = create_generator(config.generator)
        self.oracle = create_oracle(config.oracle)
        self.trainer = create_trainer(config.trainer)
        self.dynamics = create_dynamics(config.dynamics)

        self.dataset: list[Structure] = []
        self.potential: Potential | None = None

    def run(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting orchestration in {self.workdir}")

        # Initial Generation
        count = self.config.generator.get("count", 5)
        logger.info(f"Generating {count} initial structures...")

        initial_structures = list(self.generator.generate(count))
        logger.info(f"Generated {len(initial_structures)} structures.")

        if not initial_structures:
             logger.warning("No initial structures generated.")
             return

        # Initial Labeling
        logger.info("Labeling initial structures...")
        labeled_initial = list(self.oracle.compute(initial_structures))
        self.dataset.extend(labeled_initial)
        logger.info(f"Dataset size: {len(self.dataset)}")

        # Active Learning Loop
        for cycle in range(self.config.max_cycles):
            logger.info(f"--- Starting Cycle {cycle} ---")

            # Create cycle directory
            cycle_dir = self.workdir / f"cycle_{cycle}"
            cycle_dir.mkdir(parents=True, exist_ok=True)

            # Train
            logger.info("Training potential...")
            self.potential = self.trainer.train(self.dataset, cycle_dir)
            logger.info(f"Potential trained at {self.potential.path}")

            # Copy potential to match UAT expectation (flat structure)
            final_pot_path = self.workdir / f"potential_cycle_{cycle}.yace"
            if self.potential.path.exists():
                 shutil.copy(self.potential.path, final_pot_path)
                 # Update potential object to point to the new location?
                 # Or keep it pointing to the original?
                 # Let's update it so Dynamics uses the one in the main dir?
                 # Or keep using the one in cycle_dir. It doesn't matter much for Mock.
                 # But for UAT verification, the file must exist.
                 logger.info(f"Copied potential to {final_pot_path}")

            # Dynamics / Exploration
            logger.info("Running dynamics exploration...")
            candidates = list(self.dynamics.explore(self.potential))
            logger.info(f"Found {len(candidates)} candidate structures.")

            if not candidates:
                logger.info("No new candidates found. Stopping.")
                break

            # Label new candidates
            logger.info("Labeling new candidates...")
            labeled_candidates = list(self.oracle.compute(candidates))
            self.dataset.extend(labeled_candidates)
            logger.info(f"Dataset size: {len(self.dataset)}")

            logger.info(f"Cycle {cycle} complete.")
