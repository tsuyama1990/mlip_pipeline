import logging
from pathlib import Path

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.explorer = MockExplorer()
        self.oracle = MockOracle()
        self.trainer = MockTrainer()
        self.validator = MockValidator()
        self.dataset = Dataset()

    def run_loop(self) -> None:
        logger.info("Starting Active Learning Cycle")

        # Ensure work directory exists
        if not self.config.work_dir.exists():
            self.config.work_dir.mkdir(parents=True, exist_ok=True)

        potential_path: Path | None = None

        for cycle in range(1, self.config.max_cycles + 1):
            logger.info(f"--- Cycle {cycle}/{self.config.max_cycles} ---")

            # 1. Generation
            new_structures = self.explorer.generate(self.config)

            # 2. Labelling (Oracle)
            labeled_structures = self.oracle.calculate(new_structures)

            # 3. Add to Dataset
            self.dataset.structures.extend(labeled_structures)

            # 4. Training
            potential_path = self.trainer.train(self.dataset, potential_path)

            # 5. Validation
            validation_result = self.validator.validate(potential_path)

            logger.info(
                f"Cycle {cycle}/{self.config.max_cycles} completed. Validation passed: {validation_result.passed}"
            )
