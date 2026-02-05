import logging
from pathlib import Path

from config import GlobalConfig
from domain_models import Dataset
from orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator

logger = logging.getLogger("mlip_pipeline.orchestrator")


class Orchestrator:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.explorer = MockExplorer()
        self.oracle = MockOracle()
        self.trainer = MockTrainer()
        self.validator = MockValidator()
        self.dataset = Dataset(structures=[])
        self.potential_path: Path | None = None

    def run_loop(self) -> None:
        logger.info("Starting Active Learning Cycle")

        # Ensure work directory exists
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"--- Cycle {cycle}/{self.config.max_cycles} ---")

            # 1. Generation
            new_structures = self.explorer.generate(self.config)
            logger.info(f"Generated {len(new_structures)} new structures")

            # 2. Calculation (Oracle)
            labeled_structures = self.oracle.calculate(new_structures)

            # 3. Aggregate Data
            self.dataset.structures.extend(labeled_structures)

            # 4. Training
            self.potential_path = self.trainer.train(self.dataset, self.potential_path)

            # 5. Validation (Optional in loop, but good for sanity)
            validation_result = self.validator.validate(self.potential_path)
            logger.info(f"Validation result: {validation_result.passed}")

            logger.info(f"Cycle {cycle}/{self.config.max_cycles} completed")
