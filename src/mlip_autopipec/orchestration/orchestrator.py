import logging

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        # In Cycle 01, we hardcode Mock components.
        # Future cycles will use a factory pattern based on execution_mode.
        self.explorer = MockExplorer()
        self.oracle = MockOracle()
        self.trainer = MockTrainer(config.training)
        self.validator = MockValidator()

        self.dataset: list[StructureMetadata] = []
        self.current_potential: Potential | None = None

    def run(self) -> None:
        logger.info(f"Starting project: {self.config.project_name}")

        for cycle in range(1, self.config.cycles + 1):
            logger.info(f"--- Cycle {cycle}/{self.config.cycles} ---")

            # 1. Exploration
            logger.info("Phase 1: Exploration")
            candidates = self.explorer.generate_candidates(self.config.exploration)

            # 2. Oracle (Labeling)
            logger.info("Phase 2: Oracle Labeling")
            labeled_data = self.oracle.compute(candidates)
            self.dataset.extend(labeled_data)
            logger.info(f"Dataset size: {len(self.dataset)}")

            # 3. Training
            logger.info("Phase 3: Training")
            self.current_potential = self.trainer.train(self.dataset, self.current_potential)
            logger.info(f"New potential: {self.current_potential.name}")

            # 4. Validation
            logger.info("Phase 4: Validation")
            validation_result = self.validator.validate(self.current_potential)
            status = "PASSED" if validation_result.passed else "FAILED"
            logger.info(f"Validation {status}")

            if not validation_result.passed:
                logger.warning("Validation failed. Stopping loop.")
                break

        logger.info("Orchestration finished.")
