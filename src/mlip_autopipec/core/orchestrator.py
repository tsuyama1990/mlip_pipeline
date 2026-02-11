import logging
from pathlib import Path

from mlip_autopipec.components.base import (
    BaseDynamics,
    BaseGenerator,
    BaseOracle,
    BaseTrainer,
    BaseValidator,
)
from mlip_autopipec.components.mock import (
    MockDynamics,
    MockGenerator,
    MockOracle,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.core.exceptions import OrchestratorError
from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.enums import TaskStatus

logger = logging.getLogger("mlip_autopipec.orchestrator")


class Orchestrator:
    """The central controller for the MLIP pipeline."""

    def __init__(self, config: GlobalConfig, work_dir: Path):
        self.config = config
        self.work_dir = Path(work_dir)
        self.state_manager = StateManager(self.work_dir)

        # Initialize components
        self.generator = self._init_generator()
        self.oracle = self._init_oracle()
        self.trainer = self._init_trainer()
        self.dynamics = self._init_dynamics()
        self.validator = self._init_validator()

    def _init_generator(self) -> BaseGenerator:
        if self.config.generator.type == "mock":
            return MockGenerator(self.config.generator, self.work_dir / "generator")
        raise NotImplementedError(f"Generator type '{self.config.generator.type}' not implemented yet.")

    def _init_oracle(self) -> BaseOracle:
        if self.config.oracle.type == "mock":
            return MockOracle(self.config.oracle, self.work_dir / "oracle")
        raise NotImplementedError(f"Oracle type '{self.config.oracle.type}' not implemented yet.")

    def _init_trainer(self) -> BaseTrainer:
        if self.config.trainer.type == "mock":
            return MockTrainer(self.config.trainer, self.work_dir / "trainer")
        raise NotImplementedError(f"Trainer type '{self.config.trainer.type}' not implemented yet.")

    def _init_dynamics(self) -> BaseDynamics:
        if self.config.dynamics.type == "mock":
            return MockDynamics(self.config.dynamics, self.work_dir / "dynamics")
        raise NotImplementedError(f"Dynamics type '{self.config.dynamics.type}' not implemented yet.")

    def _init_validator(self) -> BaseValidator:
        if self.config.validator.type == "mock":
            return MockValidator(self.config.validator, self.work_dir / "validator")
        raise NotImplementedError(f"Validator type '{self.config.validator.type}' not implemented yet.")

    def run_loop(self) -> None:
        """Execute the main active learning loop."""
        state = self.state_manager.load()

        logger.info(f"Starting pipeline from iteration {state.current_iteration}")

        while state.current_iteration < self.config.orchestrator.n_iterations:
            state.current_iteration += 1
            logger.info(f"Cycle {state.current_iteration} Started")
            state.status = TaskStatus.RUNNING
            self.state_manager.save(state)

            try:
                # 1. Generate
                logger.info("Step 1: Structure Generation")
                candidates_iter = self.generator.generate(state)

                # Materialize iterator for now, or stream?
                # Oracle.compute takes Iterator.

                # 2. Label (Oracle)
                logger.info("Step 2: Labeling (Oracle)")
                labeled_structures = self.oracle.compute(candidates_iter)

                # Materialize to save to file (Train Dataset)
                # In a real scenario, we append to a global dataset.
                # For Cycle 01 mock, we just consume iterator and maybe count.
                # Memory: "During the Labeling stage, the Orchestrator materializes the oracle.compute iterator and writes the resulting structures to an extended XYZ file (train.xyz) in the iteration directory."

                iteration_dir = self.work_dir / f"iter_{state.current_iteration:03d}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                dataset_path = iteration_dir / "train.xyz"

                from ase.io import write
                labeled_list = list(labeled_structures)

                if labeled_list:
                    # Convert to ASE atoms
                    ase_atoms = [s.to_ase() for s in labeled_list]
                    write(dataset_path, ase_atoms)
                    state.current_dataset_path = dataset_path
                else:
                    logger.warning("No candidates generated/labeled.")

                # 3. Train
                logger.info("Step 3: Training")
                if state.current_dataset_path and state.current_dataset_path.exists():
                    training_result = self.trainer.train(
                        state.current_dataset_path,
                        state.current_potential_path
                    )
                    state.current_potential_path = training_result.potential_path
                    # Append metrics to history
                    state.history.append({
                        "iteration": state.current_iteration,
                        "metrics": training_result.metrics
                    })
                else:
                    logger.warning("Skipping training (no dataset).")

                # 4. Verify (Validator)
                if state.current_potential_path:
                    logger.info("Step 4: Validation")
                    metrics = self.validator.validate(state.current_potential_path)
                    logger.info(f"Validation Metrics: {metrics}")

                # 5. Dynamics (Explore) - Optional for Cycle 01 main flow but good to call
                # In full loop, this produces halts for next cycle.
                # For now, we just run it to verify it works.
                if state.current_potential_path:
                     logger.info("Step 5: Dynamics Exploration")
                     # We need an initial structure. Pick one from labeled set or random?
                     if labeled_list:
                         initial_struct = labeled_list[0]
                         halts = list(self.dynamics.explore(state.current_potential_path, initial_struct))
                         logger.info(f"Dynamics found {len(halts)} halts.")

                state.status = TaskStatus.COMPLETED
                logger.info(f"Cycle {state.current_iteration} Completed")
                self.state_manager.save(state)

            except Exception as e:
                state.status = TaskStatus.FAILED
                self.state_manager.save(state)
                logger.error(f"Cycle {state.current_iteration} Failed: {e}")
                if not self.config.orchestrator.continue_on_error:
                    raise OrchestratorError(f"Pipeline failed at iteration {state.current_iteration}") from e
