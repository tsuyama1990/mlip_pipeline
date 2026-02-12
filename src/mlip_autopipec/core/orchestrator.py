import logging
from collections.abc import Iterator

from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.datastructures import Structure
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
            self.state_manager.update_step(TaskType.EXPLORATION)
            self.state_manager.save()

            # Context can be expanded later
            context = {"cycle": cycle, "temperature": self.config.dynamics.temperature}
            candidates_iter = self.generator.explore(context)
            logger.info("Generated candidates iterator.")

            # 2. Oracle (Compute Labels)
            logger.info("Phase: Oracle Computing")
            self.state_manager.update_step(TaskType.TRAINING)
            self.state_manager.save()

            # Oracle returns Iterator[Structure]
            labeled_structures_iter = self.oracle.compute(candidates_iter)

            # Note: We need to consume this for training.
            # If we were using Dataset container, we'd fill it here.
            # But we are streaming.
            # However, for dynamics later, we need an initial structure.
            # We can peek/tee, or just store the first one if we want.
            # Or, Trainer returns statistics.

            # Since mock dynamics needs a structure, and we are streaming training,
            # we should probably just pick one (maybe valid one).
            # For Cycle 01 simplicity with streaming:
            # We'll collect them into a list for now to satisfy the logical need
            # to pick one for dynamics, OR we rely on Trainer to give us one? No.
            # Let's collect them into a Dataset object here (in memory) if small enough,
            # or just take the first one.
            # Given we want to fix OOM, collecting into list is bad if huge.
            # But for Cycle 01 Mock, it's small.
            # BUT the mandate is "No loading entire datasets into memory".
            # So we cannot collect all into a list.

            # We will tee the iterator? Or just use a generated structure for dynamics?
            # Or we assume Training saves data to disk and we pick from disk.
            # For now, let's create a list but limited size, or just pass iter to Trainer.

            # Problem: We need a structure for Dynamics.
            # Solution: We can implement a "tee" or custom iterator that captures the first item.

            first_structure = None

            def capture_first(iterator: Iterator[Structure]) -> Iterator[Structure]:
                nonlocal first_structure
                for i, item in enumerate(iterator):
                    if i == 0:
                        first_structure = item
                    yield item

            captured_iter = capture_first(labeled_structures_iter)

            # 3. Training
            logger.info("Phase: Training")
            self.state_manager.update_step(TaskType.TRAINING)
            self.state_manager.save()

            potential = self.trainer.train(captured_iter)
            self.state_manager.update_potential(potential.path)
            # update_potential sets dirty, so next save will write.
            self.state_manager.save()
            logger.info(f"Potential trained: {potential.path}")

            # 4. Dynamics / Simulation
            logger.info("Phase: Dynamics")
            self.state_manager.update_step(TaskType.DYNAMICS)
            self.state_manager.save()

            if first_structure:
                trajectory_iter = self.dynamics.simulate(potential, first_structure)
                # Consume trajectory to ensure simulation runs
                traj_count = sum(1 for _ in trajectory_iter)
                logger.info(f"Simulated trajectory with {traj_count} frames.")
            else:
                logger.warning("No structures available for dynamics simulation.")

            # 5. Validation
            logger.info("Phase: Validation")
            self.state_manager.update_step(TaskType.VALIDATION)
            self.state_manager.save()

            validation_result = self.validator.validate(potential)
            if validation_result.passed:
                logger.info("Validation PASSED.")
            else:
                logger.warning("Validation FAILED.")

        if self.config.orchestrator.cleanup_on_exit:
            self.state_manager.cleanup()

        logger.info("Orchestrator run completed.")
