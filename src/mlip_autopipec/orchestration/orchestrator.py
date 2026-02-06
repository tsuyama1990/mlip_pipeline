import logging
from pathlib import Path

from mlip_autopipec.config import GlobalConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, config: GlobalConfig, explorer: BaseExplorer, oracle: BaseOracle, trainer: BaseTrainer, validator: BaseValidator) -> None:
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator
        self.dataset = Dataset(name="training_set")
        self.current_potential: Path | None = None
        self.current_cycle = 0

    def run(self) -> None:
        logger.info("Orchestrator initialization complete. Starting Active Learning Loop.")
        for cycle in range(1, self.config.max_cycles + 1):
            self.current_cycle = cycle
            logger.info(f"--- Starting Cycle {cycle} ---")
            logger.info("Phase 1: Exploration")
            candidates = self.explorer.explore(self.current_potential, self.dataset)
            logger.info(f"Explorer produced {len(candidates)} candidates.")
            if not candidates:
                logger.warning("No candidates found. Stopping loop.")
                break
            for c in candidates:
                c.iteration = cycle
            logger.info("Phase 2: Labeling")
            labeled_structures = self.oracle.label(candidates)
            logger.info(f"Oracle labeled {len(labeled_structures)} structures.")
            self.dataset.structures.extend(labeled_structures)
            logger.info(f"Dataset size: {len(self.dataset.structures)}")
            logger.info("Phase 3: Training")
            new_potential = self.trainer.train(self.dataset, validation_set=None)
            logger.info(f"Trainer produced potential: {new_potential}")
            logger.info("Phase 4: Validation")
            validation_result = self.validator.validate(new_potential)
            logger.info(f"Validation Result: {validation_result.passed}, Metrics: {validation_result.metrics}")
            if validation_result.passed:
                self.current_potential = new_potential
                logger.info(f"Cycle {cycle} completed successfully. Potential updated.")
            else:
                logger.error(f"Cycle {cycle} failed validation.")
        logger.info("Active Learning Loop Finished.")
