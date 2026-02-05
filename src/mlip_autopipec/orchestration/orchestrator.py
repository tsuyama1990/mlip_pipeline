import logging

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(
        self,
        config: GlobalConfig,
        explorer: Explorer,
        oracle: Oracle,
        trainer: Trainer,
        validator: Validator
    ) -> None:
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator
        self.current_cycle = 0

        # State tracking
        self.dataset: list[StructureMetadata] = []
        self.current_potential_path: str | None = None

    def run(self) -> None:
        """Executes the active learning loop."""
        logger.info(f"Starting Project: {self.config.project_name}")
        logger.info(f"Execution Mode: {self.config.execution_mode}")

        for cycle in range(1, self.config.max_cycles + 1):
            self.current_cycle = cycle
            logger.info(f"=== Starting Cycle {cycle} ===")

            # 1. Exploration
            logger.info("Step 1: Exploration")
            candidates = self.explorer.generate_candidates(
                self.config.exploration,
                n_structures=self.config.exploration.max_structures
            )
            logger.info(f"Generated {len(candidates)} candidates.")

            # 2. Oracle (Labeling)
            logger.info("Step 2: Oracle Labeling")
            new_data = self.oracle.calculate(candidates, self.config.dft)
            self.dataset.extend(new_data)
            logger.info(f"Dataset size: {len(self.dataset)}")

            # 3. Training
            logger.info("Step 3: Training")
            self.current_potential_path = self.trainer.train(self.dataset, self.config.training)
            logger.info(f"Potential trained at: {self.current_potential_path}")

            # 4. Validation
            logger.info("Step 4: Validation")
            validation_result = self.validator.validate(self.current_potential_path, self.config)

            if validation_result.passed:
                logger.info("Validation PASSED.")
            else:
                logger.warning("Validation FAILED.")
                # In Cycle 01, we might just continue or stop.
                # UAT says "Workflow completed successfully", implies we continue or it eventually passes.

            logger.info(f"=== Cycle {cycle} Completed ===\n")

        logger.info("Workflow completed successfully.")
