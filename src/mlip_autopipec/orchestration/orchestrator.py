from pathlib import Path

from mlip_autopipec.config.config_model import SimulationConfig
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.utils.logging import get_logger, setup_logging


class Orchestrator:
    """
    The main orchestrator that manages the active learning loop.
    """

    def __init__(
        self,
        config: SimulationConfig,
        explorer: Explorer,
        oracle: Oracle,
        trainer: Trainer,
        validator: Validator
    ) -> None:
        """
        Initialize the Orchestrator with configuration and components.

        Args:
            config: The simulation configuration.
            explorer: The structure generator component.
            oracle: The ground truth provider component.
            trainer: The potential training component.
            validator: The validation component.
        """
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator

        # Setup logger
        setup_logging(name="mlip_autopipec")
        self.logger = get_logger("mlip_autopipec.orchestration")

        self.logger.info(f"PYACEMAKER initialized for project: {config.project_name}")
        self.logger.info("Configuration loaded successfully.")

        self.current_potential: Path | None = None

    def run_loop(self) -> None:
        """
        Execute the active learning loop.
        """
        self.logger.info(f"Starting loop for project: {self.config.project_name}")

        # Step 1: Exploration
        self.logger.info("Starting Exploration phase...")
        candidate_structures = self.explorer.explore(self.current_potential)
        self.logger.info(f"Exploration completed. Generated {len(candidate_structures)} structures.")

        # Step 2: Oracle (Labeling)
        self.logger.info("Starting Oracle phase (DFT calculations)...")
        labelled_structures = self.oracle.compute(candidate_structures)
        self.logger.info(f"DFT calculations completed. Labelled {len(labelled_structures)} structures.")

        # Step 3: Training
        self.logger.info("Starting Training phase...")
        self.current_potential = self.trainer.train(labelled_structures)
        self.logger.info(f"Training completed. New potential saved to {self.current_potential}")

        # Step 4: Validation
        self.logger.info("Starting Validation phase...")
        validation_result = self.validator.validate(self.current_potential)

        status = "PASSED" if validation_result.passed else "FAILED"
        self.logger.info(f"Validation completed. Status: {status}")

        for metric in validation_result.metrics:
            self.logger.info(f"  - {metric.name}: {metric.score} ({'PASS' if metric.passed else 'FAIL'})")
