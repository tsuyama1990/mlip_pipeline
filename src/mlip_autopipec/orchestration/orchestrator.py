import logging
from pathlib import Path

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.dataset import Dataset
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main pipeline orchestrator.
    Manages the active learning loop using injected components.
    """

    def __init__(
        self,
        config: GlobalConfig,
        explorer: Explorer,
        oracle: Oracle,
        trainer: Trainer,
        validator: Validator,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            config: Global configuration.
            explorer: Component for structure generation.
            oracle: Component for property calculation.
            trainer: Component for model training.
            validator: Component for model validation.
        """
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator
        self.dataset = Dataset()
        self.current_potential: Path | None = None

    def run_loop(self) -> None:
        """Run the active learning loop."""
        logger.info(f"Starting pipeline for {self.config.max_cycles} cycles.")

        for cycle in range(self.config.max_cycles):
            logger.info(f"--- Cycle {cycle + 1}/{self.config.max_cycles} ---")

            # 1. Explore
            structures = self.explorer.generate(self.config)
            logger.info(f"Generated {len(structures)} structures.")

            # 2. Oracle
            labeled_structures = self.oracle.calculate(structures)

            # 3. Update Dataset
            # Encapsulated access to dataset
            self.dataset.add_batch(labeled_structures)

            # 4. Train
            self.current_potential = self.trainer.train(self.dataset, self.current_potential)
            logger.info(f"Potential updated: {self.current_potential}")

            # 5. Validate
            if self.current_potential:
                result = self.validator.validate(self.current_potential)
                logger.info(f"Validation result: {result}")

        logger.info("Pipeline completed.")
