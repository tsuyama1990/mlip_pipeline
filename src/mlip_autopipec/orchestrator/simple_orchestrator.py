import logging
from collections import deque
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

        # Memory Safety: Use deque to limit dataset size
        # Hardcoded limit for now, could be in config
        self.max_dataset_size = 1000
        self.dataset: deque[Structure] = deque(maxlen=self.max_dataset_size)

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

        # Avoid hardcoded path, use config-derived or safe default
        # Assuming project_name is safe (pydantic validated to be string, but maybe check characters?)
        # For now, using Path object is better than string concatenation
        workdir_root = Path("active_learning") / self.config.project_name
        workdir_root.mkdir(parents=True, exist_ok=True)

        try:
            # Initial candidates
            logger.info("Generating initial candidates...")
            candidates = self.structure_generator.get_candidates()
            # Do NOT add to dataset yet, wait for computation
        except Exception:
            logger.exception("Failed to generate initial candidates.")
            raise

        potential: Potential | None = None

        for i in range(1, self.config.orchestrator.max_iterations + 1):
            logger.info(f"Starting Iteration {i}")
            iter_dir = workdir_root / f"iter_{i:03d}"
            iter_dir.mkdir(exist_ok=True)

            try:
                logger.info("Labeling structures...")
                # Update candidates with computed properties (new objects returned)
                computed_candidates = self.oracle.compute(candidates, workdir=iter_dir)

                # Add computed structures to dataset
                self.dataset.extend(computed_candidates)

                logger.info("Training potential...")
                # Convert deque to list for Trainer interface
                potential = self.trainer.train(list(self.dataset), workdir=iter_dir)

                logger.info("Validating potential...")
                val_result = self.validator.validate(potential, workdir=iter_dir)
                if val_result.passed:
                    logger.info("Validation passed.")
                    break

                logger.info("Validation failed. Proceeding to exploration.")

                logger.info("Running exploration...")
                exploration_result = self.dynamics.run_exploration(
                    potential, workdir=iter_dir
                )

                if exploration_result.halted:
                    logger.info("Exploration halted. New candidates found.")
                    candidates = exploration_result.structures
                    # Loop continues, these candidates will be computed in next iteration
                else:
                    logger.info("Exploration converged.")
                    break
            except Exception:
                logger.exception(f"Error during iteration {i}")
                raise

        logger.info("Active Learning Cycle Completed Successfully")
