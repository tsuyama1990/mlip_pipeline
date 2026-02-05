import logging

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.interfaces.core_interfaces import (
    Explorer,
    Oracle,
    Trainer,
    Validator,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        explorer: Explorer,
        oracle: Oracle,
        trainer: Trainer,
        validator: Validator,
        config: GlobalConfig,
    ) -> None:
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator
        self.config = config

    def run(self) -> None:
        logger.info("Starting Orchestrator Loop")

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"Starting Cycle {cycle}")

            # 1. Exploration
            candidates = self.explorer.generate_candidates(self.config.exploration)
            logger.info(f"Cycle {cycle}: Generated {len(candidates)} candidates")

            # 2. Oracle (Labeling)
            labeled_structures = self.oracle.calculate(candidates, self.config.dft)
            logger.info(f"Cycle {cycle}: Labeled {len(labeled_structures)} structures")

            # 3. Training
            potential_path = self.trainer.train(labeled_structures, self.config.training)
            logger.info(f"Cycle {cycle}: Trained potential at {potential_path}")

            # 4. Validation
            validation_result = self.validator.validate(potential_path)
            status = "PASSED" if validation_result.passed else "FAILED"
            logger.info(f"Cycle {cycle}: Validation {status}")

        logger.info("Workflow completed successfully")
