import logging
from collections.abc import Iterator
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
from mlip_autopipec.domain_models.inputs import ProjectState, Structure

logger = logging.getLogger("mlip_autopipec.orchestrator")


class Orchestrator:
    """The central controller for the MLIP pipeline."""

    def __init__(self, config: GlobalConfig, work_dir: Path) -> None:
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
        msg = f"Generator type '{self.config.generator.type}' not implemented yet."
        raise NotImplementedError(msg)

    def _init_oracle(self) -> BaseOracle:
        if self.config.oracle.type == "mock":
            return MockOracle(self.config.oracle, self.work_dir / "oracle")
        msg = f"Oracle type '{self.config.oracle.type}' not implemented yet."
        raise NotImplementedError(msg)

    def _init_trainer(self) -> BaseTrainer:
        if self.config.trainer.type == "mock":
            return MockTrainer(self.config.trainer, self.work_dir / "trainer")
        msg = f"Trainer type '{self.config.trainer.type}' not implemented yet."
        raise NotImplementedError(msg)

    def _init_dynamics(self) -> BaseDynamics:
        if self.config.dynamics.type == "mock":
            return MockDynamics(self.config.dynamics, self.work_dir / "dynamics")
        msg = f"Dynamics type '{self.config.dynamics.type}' not implemented yet."
        raise NotImplementedError(msg)

    def _init_validator(self) -> BaseValidator:
        if self.config.validator.type == "mock":
            return MockValidator(self.config.validator, self.work_dir / "validator")
        msg = f"Validator type '{self.config.validator.type}' not implemented yet."
        raise NotImplementedError(msg)

    def run_loop(self) -> None:
        """Execute the main active learning loop."""
        state = self.state_manager.load()

        logger.info(f"Starting pipeline from iteration {state.current_iteration}")

        while state.current_iteration < self.config.orchestrator.n_iterations:
            try:
                self._run_iteration(state)
            except Exception as e:
                state.status = TaskStatus.FAILED
                self.state_manager.save(state)
                logger.exception(f"Cycle {state.current_iteration} Failed")
                if not self.config.orchestrator.continue_on_error:
                    msg = f"Pipeline failed at iteration {state.current_iteration}"
                    raise OrchestratorError(msg) from e

    def _run_iteration(self, state: ProjectState) -> None:
        state.current_iteration += 1
        logger.info(f"Cycle {state.current_iteration} Started")
        state.status = TaskStatus.RUNNING
        self.state_manager.save(state)

        # Import locally to avoid top-level dependency issues
        from ase.io import write

        candidates_iter: Iterator[Structure]
        is_cold_start = state.current_potential_path is None or not state.current_potential_path.exists()

        if is_cold_start:
            logger.info("Mode: Cold Start (Global Search)")
            candidates_iter = self.generator.generate(state)
        else:
            logger.info("Mode: Active Learning (Dynamics & Diagnosis)")
            # 1. Exploration (Dynamics)
            # We need a seed structure. Use the last structure from training data or random?
            # For simplicity, we assume we can pick one from the previous dataset or generator.
            # In a real scenario, this would be more robust.
            seed_structure = self._get_seed_structure(state)

            logger.info("Step 1: Dynamics Exploration")
            # This yields "Halt" structures
            halts = self.dynamics.explore(state.current_potential_path, seed_structure) # type: ignore[arg-type]

            # 2. Selection (Local Generation & D-Optimality)
            logger.info("Step 2: Diagnosis & Local Selection")
            local_candidates_pool: list[Structure] = []
            for halt in halts:
                # Generate local candidates around the halt
                locals_ = self.generator.generate_local(halt, n_candidates=20)
                # Select best ones (D-Optimality)
                selected_locals = self.trainer.select_local_active_set(locals_, n_selection=5)
                local_candidates_pool.extend(selected_locals)

            candidates_iter = iter(local_candidates_pool)

        # 3. Labeling (Oracle)
        logger.info("Step 3: Labeling (Oracle)")
        labeled_structures = self.oracle.compute(candidates_iter)
        labeled_list = list(labeled_structures)

        if not labeled_list:
            logger.warning("No candidates generated/labeled.")
            # If no candidates, we might want to skip training, but for now we proceed

        # Save labeled data
        iteration_dir = self.work_dir / f"iter_{state.current_iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        new_dataset_path = iteration_dir / "train.xyz"

        ase_atoms = [s.to_ase() for s in labeled_list]
        write(new_dataset_path, ase_atoms)

        # Update cumulative dataset (simplistic approach: just point to new one for now)
        # In real system, we would merge datasets.
        # For Mock, we just use the new one.
        state.current_dataset_path = new_dataset_path

        # 4. Training (Refinement)
        logger.info("Step 4: Training")
        if state.current_dataset_path and state.current_dataset_path.exists():
            training_result = self.trainer.train(
                state.current_dataset_path,
                state.current_potential_path
            )
            state.current_potential_path = training_result.potential_path
            state.history.append({
                "iteration": state.current_iteration,
                "metrics": training_result.metrics
            })
        else:
            logger.warning("Skipping training (no dataset).")

        # 5. Validation
        if state.current_potential_path:
            logger.info("Step 5: Validation")
            metrics = self.validator.validate(state.current_potential_path)
            logger.info(f"Validation Metrics: {metrics}")

        state.status = TaskStatus.COMPLETED
        logger.info(f"Cycle {state.current_iteration} Completed")
        self.state_manager.save(state)

    def _get_seed_structure(self, state: ProjectState) -> Structure:
        """Get a seed structure for dynamics."""
        # Try to load from current dataset
        from ase.io import read
        if state.current_dataset_path and state.current_dataset_path.exists():
            atoms = read(state.current_dataset_path, index=0) # Take first
            return Structure.from_ase(atoms) # type: ignore[arg-type]

        # Fallback: Generate a random one
        # This is a bit of a hack for the Mock, but necessary if no data exists yet (shouldn't happen if Cold Start ran)
        return next(self.generator.generate(state))
