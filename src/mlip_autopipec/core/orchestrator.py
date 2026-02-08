import logging
import shutil

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models import GlobalConfig, Potential
from mlip_autopipec.factory import create_component
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseGenerator,
    BaseOracle,
    BaseTrainer,
    BaseValidator,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Manages the active learning pipeline loop.
    """

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.workdir = config.workdir
        self.dataset = Dataset(self.workdir / config.dataset_filename)

        # Instantiate components
        self.generator: BaseGenerator = create_component("generator", config.generator)
        self.oracle: BaseOracle = create_component("oracle", config.oracle)
        self.trainer: BaseTrainer = create_component("trainer", config.trainer)
        self.dynamics: BaseDynamics = create_component("dynamics", config.dynamics)
        self.validator: BaseValidator | None = None
        if config.validator:
            self.validator = create_component("validator", config.validator)

    def run(self) -> None:
        """
        Executes the pipeline cycles.
        """
        logger.info(f"Starting pipeline in {self.workdir}")
        self.workdir.mkdir(parents=True, exist_ok=True)

        potential: Potential | None = None

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"=== Starting Cycle {cycle} ===")

            # 1. Generation / Exploration
            if potential is None:
                # First cycle: use Generator
                logger.info("Cycle 1: Using Generator for initial structures.")
                candidates = self.generator.generate(potential=None)
            else:
                # Subsequent cycles: use Dynamics (Active Learning)
                logger.info(f"Cycle {cycle}: Using Dynamics for exploration.")
                candidates = self.dynamics.run(potential=potential)

            # 2. Labelling (Oracle)
            # Stream candidates -> oracle -> dataset
            logger.info("Oracle: Computing labels...")
            labeled_structures = self.oracle.compute(candidates)

            # 3. Update Dataset
            logger.info("Dataset: Appending new structures...")
            count = self.dataset.append(labeled_structures)
            logger.info(f"Dataset: Added {count} structures. Total: {len(self.dataset)}")

            if count == 0:
                logger.warning("No new structures added in this cycle.")

            # 4. Training
            logger.info("Trainer: Training potential...")
            potential = self.trainer.train(
                dataset=self.dataset,
                initial_potential=potential,
                workdir=self.workdir,
            )

            # Save/Rename potential artifact
            target_filename = f"potential_cycle_{cycle}{self.config.potential_extension}"
            target_path = self.workdir / target_filename

            if potential.path != target_path and potential.path.exists():
                # Copy to target path
                logger.info(f"Saving potential to {target_path}")
                shutil.copy(potential.path, target_path)
                potential.path = target_path

            # 5. Validation
            if self.validator:
                logger.info("Validator: Validating potential...")
                metrics = self.validator.validate(potential)
                logger.info(f"Validation Metrics: {metrics}")

            logger.info(f"=== Completed Cycle {cycle} ===")
