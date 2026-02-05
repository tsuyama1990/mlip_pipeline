import logging

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.dataset import Dataset
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

        # Use the Dataset abstraction which supports future lazy loading
        self.dataset = Dataset(name="main_dataset")
        self.current_cycle = 0

    def run(self) -> None:
        """
        Execute the active learning loop.
        """
        logger.info("Starting Orchestrator loop...")

        cycles = self.config.max_cycles

        for i in range(cycles):
            self.current_cycle = i
            logger.info(f"--- Cycle {i+1}/{cycles} ---")

            # 1. Exploration: Generate candidates
            candidates = self.explorer.generate_candidates()
            logger.info(f"Generated {len(candidates)} candidates.")

            if not candidates:
                logger.warning("No candidates generated. Stopping loop.")
                break

            # 2. Oracle: Calculate properties (Batch processing)
            labeled = self.oracle.calculate(candidates)
            logger.info(f"Calculated {len(labeled)} structures.")

            # 3. Add to Dataset (Active Set Selection would happen here or inside Dataset)
            # For Cycle 01, we add all. The Dataset class handles storage efficiency.
            for s in labeled:
                self.dataset.add(s)

            # 4. Training
            potential_path = self.trainer.train(self.dataset)
            logger.info(f"Trained potential: {potential_path}")

            # 5. Validation
            val_result = self.validator.validate(potential_path)
            if val_result.passed:
                logger.info("Validation PASSED.")
            else:
                logger.warning("Validation FAILED.")
                # In future cycles, we might halt or adjust strategy here

        logger.info("Orchestrator loop finished.")
