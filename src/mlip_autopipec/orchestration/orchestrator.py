import logging

from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.domain_models.dynamics import MDState
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.orchestration.mocks import (
    MockExplorer,
    MockOracle,
    MockTrainer,
    MockValidator,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Manages the active learning workflow."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        # For Cycle 01, we default to Mocks.
        # In future cycles, this will choose implementations based on config.
        self.explorer: Explorer = MockExplorer()
        self.oracle: Oracle = MockOracle()
        self.trainer: Trainer = MockTrainer()
        self.validator: Validator = MockValidator()
        self.state: MDState | None = None

    def run(self) -> None:
        """Execute the orchestration loop."""
        logger.info("Starting orchestration cycle...")

        # Phase 1: Exploration
        logger.info("Phase 1: Exploration")
        structures = self.explorer.explore(self.state)
        logger.info(f"Generated {len(structures)} structures")

        # Phase 2: Oracle
        logger.info("Phase 2: Oracle Labeling")
        labeled_structures = self.oracle.compute(structures)

        # Phase 3: Training
        logger.info("Phase 3: Training")
        potential_path = self.trainer.train(labeled_structures)

        # Phase 4: Validation
        logger.info("Phase 4: Validation")
        result = self.validator.validate(potential_path)

        if result.passed:
            logger.info("Cycle completed successfully")
        else:
            logger.warning("Cycle failed validation")
