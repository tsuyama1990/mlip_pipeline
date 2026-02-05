import logging
from pathlib import Path

from src.config.config_model import GlobalConfig
from src.domain_models import Dataset
from src.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: GlobalConfig,
        explorer: Explorer,
        oracle: Oracle,
        trainer: Trainer,
        validator: Validator,
    ) -> None:
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator
        self.dataset = Dataset()
        self.current_potential: Path | None = None

    def run(self) -> None:
        logger.info("Starting Active Learning Cycle")
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        for cycle in range(1, self.config.max_cycles + 1):
            cycle_dir = self.config.work_dir / f"cycle_{cycle:02d}"
            cycle_dir.mkdir(exist_ok=True)
            logger.info(f"--- Starting Cycle {cycle}/{self.config.max_cycles} ---")

            # 1. Generate
            candidates = self.explorer.generate(self.config)
            logger.info(f"Generated {len(candidates)} candidates")

            # 2. Oracle
            labeled = self.oracle.calculate(candidates)
            for item in labeled:
                self.dataset.add(item)

            # 3. Train
            new_potential = self.trainer.train(self.dataset, self.current_potential)
            self.current_potential = new_potential

            # 4. Validate
            validation_result = self.validator.validate(self.current_potential)
            if not validation_result.passed:
                logger.warning(f"Cycle {cycle} validation failed: {validation_result.metrics}")
            else:
                logger.info(f"Cycle {cycle} validation passed")

            logger.info(f"Cycle {cycle}/{self.config.max_cycles} completed")
