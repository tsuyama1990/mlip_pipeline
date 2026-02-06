import logging
from pathlib import Path

from mlip_autopipec.config import GlobalConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(
        self,
        config: GlobalConfig,
        explorer: BaseExplorer,
        oracle: BaseOracle,
        trainer: BaseTrainer,
        validator: BaseValidator # Added Validator as per implication
    ) -> None:
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator

        # In a real scenario, we might load an initial dataset.
        # For Cycle 01, we start empty or with what Explorer gives.
        self.dataset = Dataset(structures=[])
        self.current_potential_path = Path("initial_potential.yace") # Placeholder

    def run(self) -> None:
        logger.info("Orchestrator initialization complete")

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"Starting Cycle {cycle}...")

            # 1. Explore
            logger.info("Running Explorer...")
            new_candidates = self.explorer.explore(self.current_potential_path, self.dataset)
            logger.info(f"Explorer produced {len(new_candidates.structures)} candidates.")

            # 2. Oracle
            logger.info("Running Oracle...")
            labeled_data = self.oracle.label(new_candidates)
            logger.info(f"Oracle labeled {len(labeled_data.structures)} structures.")

            # Accumulate data
            self.dataset.structures.extend(labeled_data.structures)

            # 3. Train
            logger.info("Running Trainer...")
            # For simplicity in Cycle 01, we use the whole dataset as train and valid
            # In real AL, we split.
            potential_path = self.trainer.train(self.dataset, self.dataset)
            self.current_potential_path = potential_path
            logger.info(f"Trainer produced potential version {cycle} at {potential_path}.")

            # 4. Validate (Optional but good)
            validation_result = self.validator.validate(potential_path)
            logger.info(f"Validation Result: {validation_result}")

            logger.info(f"Cycle {cycle} completed.")

        logger.info("All cycles finished.")
