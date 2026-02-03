from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.utils.logging import get_logger

logger = get_logger(__name__)

class Orchestrator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.logger = logger

        # Instantiate components (Mocks for Cycle 01)
        # In future cycles, this will use a Factory based on config
        self.explorer: Explorer = MockExplorer()
        self.oracle: Oracle = MockOracle()
        self.trainer: Trainer = MockTrainer()
        self.validator: Validator = MockValidator()

        self.logger.info(f"PYACEMAKER initialized for project: {config.project_name}")
        self.logger.debug(f"Configuration loaded: {config.model_dump_json()}")

    def run(self) -> None:
        """Runs the active learning loop."""
        self.logger.info("Starting orchestration cycle...")

        try:
            # 1. Exploration
            self.logger.info("Phase 1: Exploration")
            structures = self.explorer.explore(current_potential=None)
            self.logger.info(f"Generated {len(structures)} candidate structures")

            # 2. Labeling (Oracle)
            self.logger.info("Phase 2: Oracle Labeling")
            labelled_data = self.oracle.compute(structures)
            self.logger.info(f"Labelled {len(labelled_data)} structures")

            # 3. Training
            self.logger.info("Phase 3: Training")
            potential_path = self.trainer.train(labelled_data)
            self.logger.info(f"Potential trained at {potential_path}")

            # 4. Validation
            self.logger.info("Phase 4: Validation")
            report = self.validator.validate(potential_path)

            if report.passed:
                self.logger.info("Validation PASSED")
            else:
                self.logger.warning("Validation FAILED")

            self.logger.info("Cycle completed successfully.")

        except Exception:
            self.logger.exception("Orchestration failed with error")
            raise
