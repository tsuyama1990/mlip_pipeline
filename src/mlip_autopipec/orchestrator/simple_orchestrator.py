import logging
from pathlib import Path

from mlip_autopipec.config.base_config import GlobalConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockOracle,
    MockStructureGenerator,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseOrchestrator,
    BaseStructureGenerator,
    BaseTrainer,
    BaseValidator,
)

logger = logging.getLogger(__name__)


class SimpleOrchestrator(BaseOrchestrator):
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.oracle: BaseOracle
        self.trainer: BaseTrainer
        self.dynamics: BaseDynamics
        self.structure_generator: BaseStructureGenerator
        self.validator: BaseValidator
        self.dataset: list[Structure] = []

        self._initialize_components()

    def _initialize_components(self) -> None:
        # Oracle
        if self.config.oracle.type == "mock":
            self.oracle = MockOracle(self.config.oracle.params)
        else:
            msg = f"Oracle type {self.config.oracle.type} not implemented"
            raise NotImplementedError(msg)

        # Trainer
        if self.config.trainer.type == "mock":
            self.trainer = MockTrainer(self.config.trainer.params)
        else:
            msg = f"Trainer type {self.config.trainer.type} not implemented"
            raise NotImplementedError(msg)

        # Dynamics
        if self.config.dynamics.type == "mock":
            self.dynamics = MockDynamics(self.config.dynamics.params)
        else:
            msg = f"Dynamics type {self.config.dynamics.type} not implemented"
            raise NotImplementedError(msg)

        # Structure Generator
        if self.config.structure_generator.type == "mock":
            self.structure_generator = MockStructureGenerator(
                self.config.structure_generator.params
            )
        else:
            msg = f"Structure Generator type {self.config.structure_generator.type} not implemented"
            raise NotImplementedError(msg)

        # Validator
        if self.config.validator.type == "mock":
            self.validator = MockValidator(self.config.validator.params)
        else:
            msg = f"Validator type {self.config.validator.type} not implemented"
            raise NotImplementedError(msg)

    def run(self) -> None:
        logger.info("Starting Active Learning Cycle")
        workdir_root = Path("active_learning")
        workdir_root.mkdir(exist_ok=True)

        # Initial candidates
        candidates = self.structure_generator.get_candidates()
        self.dataset.extend(candidates)

        potential: Potential | None = None

        for i in range(1, self.config.orchestrator.max_iterations + 1):
            logger.info(f"Starting Iteration {i}")
            iter_dir = workdir_root / f"iter_{i:03d}"
            iter_dir.mkdir(exist_ok=True)

            logger.info("Labeling structures...")
            # For Cycle 01, assume we re-label everything or label just candidates
            # Here we label the 'candidates' variable which holds new structures
            self.oracle.compute(candidates, workdir=iter_dir)

            logger.info("Training potential...")
            potential = self.trainer.train(self.dataset, workdir=iter_dir)

            logger.info("Validating potential...")
            val_result = self.validator.validate(potential, workdir=iter_dir)
            if val_result.passed:
                logger.info("Validation passed.")
                # For mock cycle, we might want to continue unless explicitly told to stop
                # But SPEC says terminate.
                break

            logger.info("Running exploration...")
            exploration_result = self.dynamics.run_exploration(
                potential, workdir=iter_dir
            )

            if exploration_result.halted:
                logger.info("Exploration halted. Adding new structures to dataset.")
                candidates = exploration_result.structures
                self.dataset.extend(candidates)
            else:
                logger.info("Exploration converged.")
                # Implicitly break if converged?
                break

        logger.info("Active Learning Cycle Completed Successfully")
