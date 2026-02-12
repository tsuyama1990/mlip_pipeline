import logging

from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TaskType,
    TrainerType,
    ValidatorType,
)
from mlip_autopipec.dynamics import BaseDynamics, MockDynamics
from mlip_autopipec.generator import BaseGenerator, MockGenerator
from mlip_autopipec.oracle import BaseOracle, MockOracle
from mlip_autopipec.trainer import BaseTrainer, MockTrainer
from mlip_autopipec.validator import BaseValidator, MockValidator

logger = logging.getLogger(__name__)


class Orchestrator:
    """The central controller for the Active Learning Pipeline."""

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.work_dir = config.orchestrator.work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.state_manager = StateManager(self.work_dir)

        # Initialize components
        self.generator = self._create_generator()
        self.oracle = self._create_oracle()
        self.trainer = self._create_trainer()
        self.dynamics = self._create_dynamics()
        self.validator = self._create_validator()

    def _create_generator(self) -> BaseGenerator:
        if self.config.generator.type == GeneratorType.MOCK:
            return MockGenerator()
        msg = f"Unsupported generator type: {self.config.generator.type}"
        raise ValueError(msg)

    def _create_oracle(self) -> BaseOracle:
        if self.config.oracle.type == OracleType.MOCK:
            return MockOracle()
        msg = f"Unsupported oracle type: {self.config.oracle.type}"
        raise ValueError(msg)

    def _create_trainer(self) -> BaseTrainer:
        if self.config.trainer.type == TrainerType.MOCK:
            return MockTrainer(self.work_dir)
        msg = f"Unsupported trainer type: {self.config.trainer.type}"
        raise ValueError(msg)

    def _create_dynamics(self) -> BaseDynamics:
        if self.config.dynamics.type == DynamicsType.MOCK:
            return MockDynamics()
        msg = f"Unsupported dynamics type: {self.config.dynamics.type}"
        raise ValueError(msg)

    def _create_validator(self) -> BaseValidator:
        if self.config.validator.type == ValidatorType.MOCK:
            return MockValidator()
        msg = f"Unsupported validator type: {self.config.validator.type}"
        raise ValueError(msg)

    def run(self) -> None:
        """Executes the active learning workflow."""
        logger.info("Starting Orchestrator...")

        max_cycles = self.config.orchestrator.max_cycles
        start_cycle = self.state_manager.state.current_cycle

        for cycle in range(start_cycle, max_cycles):
            self.state_manager.update_cycle(cycle + 1)
            logger.info(f"--- Starting Cycle {cycle + 1}/{max_cycles} ---")

            # 1. Exploration / Generation
            logger.info("Phase: Exploration")
            self.state_manager.state.current_step = TaskType.EXPLORATION
            self.state_manager.save()

            # Context can be expanded later
            context = {"cycle": cycle, "temperature": self.config.dynamics.temperature}
            candidates = self.generator.explore(context)
            logger.info(f"Generated {len(candidates)} candidates.")

            # 2. Oracle (Compute Labels)
            logger.info("Phase: Oracle Computing")
            self.state_manager.state.current_step = TaskType.TRAINING  # Or keep as Oracle
            # Note: TaskType has EXPLORATION, TRAINING, DYNAMICS, VALIDATION.
            # Oracle is technically part of data gen or training prep.
            self.state_manager.save()

            labeled_dataset = self.oracle.compute(candidates)
            logger.info(f"Computed labels for {len(labeled_dataset)} structures.")

            # 3. Training
            logger.info("Phase: Training")
            self.state_manager.state.current_step = TaskType.TRAINING
            self.state_manager.save()

            potential = self.trainer.train(labeled_dataset)
            self.state_manager.update_potential(potential.path)
            logger.info(f"Potential trained: {potential.path}")

            # 4. Dynamics / Simulation (Optional loop check, but here sequential)
            # In Cycle 01, we just run dynamics as a test or "Simulate" phase?
            # SPEC says: "Dynamics Engine... executes simulations using the trained potential."
            # And "Uncertainty Watchdog... halts simulations".
            # For Cycle 01 mock, we can just run it once per cycle to simulate usage.
            logger.info("Phase: Dynamics")
            self.state_manager.state.current_step = TaskType.DYNAMICS
            self.state_manager.save()

            # Use the first structure from dataset as initial for now
            if len(labeled_dataset) > 0:
                initial_structure = labeled_dataset.structures[0]
                trajectory = self.dynamics.simulate(potential, initial_structure)
                logger.info(f"Simulated trajectory with {len(trajectory.structures)} frames.")

            # 5. Validation
            logger.info("Phase: Validation")
            self.state_manager.state.current_step = TaskType.VALIDATION
            self.state_manager.save()

            validation_result = self.validator.validate(potential)
            if validation_result.passed:
                logger.info("Validation PASSED.")
            else:
                logger.warning("Validation FAILED.")

        if self.config.orchestrator.cleanup_on_exit:
            self.state_manager.cleanup()

        logger.info("Orchestrator run completed.")
