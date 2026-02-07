import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models import (
    DynamicsConfig,
    ExplorationResult,
    GeneratorConfig,
    GlobalConfig,
    OracleConfig,
    SelectorConfig,
    Structure,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockOracle,
    MockSelector,
    MockStructureGenerator,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseSelector,
    BaseStructureGenerator,
    BaseTrainer,
    BaseValidator,
)

logger = logging.getLogger(__name__)

class SimpleOrchestrator:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.workdir = config.workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.oracle = self._create_oracle(config.oracle)
        self.trainer = self._create_trainer(config.trainer)
        self.dynamics = self._create_dynamics(config.dynamics)
        self.generator = self._create_generator(config.generator)
        self.validator = self._create_validator(config.validator)
        self.selector = self._create_selector(config.selector)

        # State: File-based persistence
        self.dataset_path = self.workdir / "dataset.jsonl"
        self.cycle_count = 0

        if config.initial_structure_path and config.initial_structure_path.exists():
             logger.warning(f"Loading from {config.initial_structure_path} not implemented in Cycle 01. Using dummy.")
             self.initial_structure = self._create_dummy_structure()
        else:
             self.initial_structure = self._create_dummy_structure()

    def _create_dummy_structure(self) -> Structure:
        return Structure(
            positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            cell=np.eye(3) * 10.0,
            species=["Ar", "Ar"]
        )

    def _create_oracle(self, config: OracleConfig) -> BaseOracle:
        if config.type == "mock":
            return MockOracle(config.params)
        msg = f"Unknown oracle type: {config.type}"
        raise ValueError(msg)

    def _create_trainer(self, config: TrainerConfig) -> BaseTrainer:
        if config.type == "mock":
            return MockTrainer(config.params)
        msg = f"Unknown trainer type: {config.type}"
        raise ValueError(msg)

    def _create_dynamics(self, config: DynamicsConfig) -> BaseDynamics:
        if config.type == "mock":
            return MockDynamics(config.params)
        msg = f"Unknown dynamics type: {config.type}"
        raise ValueError(msg)

    def _create_generator(self, config: GeneratorConfig) -> BaseStructureGenerator:
        if config.type in {"mock", "random"}:
            return MockStructureGenerator(config.params)
        msg = f"Unknown generator type: {config.type}"
        raise ValueError(msg)

    def _create_validator(self, config: ValidatorConfig) -> BaseValidator:
        if config.type == "mock":
            return MockValidator(config.params)
        msg = f"Unknown validator type: {config.type}"
        raise ValueError(msg)

    def _create_selector(self, config: SelectorConfig) -> BaseSelector:
        if config.type in {"mock", "random"}:
            return MockSelector(config.params)
        msg = f"Unknown selector type: {config.type}"
        raise ValueError(msg)

    def run(self) -> None:
        logger.info("Starting Orchestrator loop...")

        current_structure = self.initial_structure
        max_cycles = self.config.max_cycles

        for i in range(max_cycles):
            self.cycle_count = i + 1
            msg = f"--- Cycle {self.cycle_count} started ---"
            logger.info(msg)

            candidates = self._step_exploration(current_structure)

            # Selection step typically needs awareness of existing data.
            # For scalability, we might just pass a count or metadata,
            # but the interface accepts list[Structure].
            # We assume candidates are in memory (generated batch).
            # The 'existing_data' is optional in interface. For now, we pass None to avoid loading.
            selected_candidates = self._step_selection(candidates)

            new_data_iterator = self._step_calculation(selected_candidates)

            # Persist new data immediately (Streaming)
            last_structure = None
            count = 0
            for struct in new_data_iterator:
                self._append_structure(struct)
                last_structure = struct
                count += 1

            if count == 0:
                logger.warning("No new data from Oracle. Stopping.")
                break

            logger.info(f"Added {count} structures to dataset at {self.dataset_path}")

            potential_path = self._step_refinement()
            self._step_validation(potential_path)

            # Use the last computed structure for dynamics
            if last_structure:
                dynamics_result = self._step_deployment(potential_path, last_structure)
                if dynamics_result.trajectory:
                    current_structure = dynamics_result.trajectory[-1]

                if dynamics_result.status == "halted":
                     logger.info("Dynamics halted. Stopping loop.")
                     break

            msg = f"--- Cycle {self.cycle_count} completed ---"
            logger.info(msg)

        logger.info("Orchestrator finished.")

    def _append_structure(self, structure: Structure) -> None:
        """
        Append a structure to the file-based dataset.
        """
        # Append as JSONL
        with self.dataset_path.open("a") as f:
            f.write(structure.model_dump_json() + "\n")

    def _iterate_dataset(self) -> Iterator[Structure]:
        """
        Yield structures from the file-based dataset.
        """
        if not self.dataset_path.exists():
            return

        with self.dataset_path.open("r") as f:
            for line in f:
                if line.strip():
                    yield Structure.model_validate_json(line)

    def _step_exploration(self, current_structure: Structure) -> list[Structure]:
        try:
            candidates = self.generator.generate(current_structure, strategy="random")
            logger.info(f"Generated {len(candidates)} candidates")
        except Exception as e:
            logger.exception("Generator failed unexpectedly")
            msg = f"Cycle {self.cycle_count} failed at Generator stage"
            raise RuntimeError(msg) from e
        else:
            return candidates

    def _step_selection(self, candidates: list[Structure]) -> list[Structure]:
        try:
            n_select = self.config.selector.params.get("n", 5)
            # Pass None for existing_data to avoid OOM
            selected_candidates = self.selector.select(candidates, n=n_select, existing_data=None)
            logger.info(f"Selected {len(selected_candidates)} candidates for labeling")
        except Exception as e:
            logger.exception("Selector failed unexpectedly")
            msg = f"Cycle {self.cycle_count} failed at Selector stage"
            raise RuntimeError(msg) from e
        else:
            return selected_candidates

    def _step_calculation(self, selected_candidates: list[Structure]) -> Iterator[Structure]:
        try:
            return self.oracle.compute_batch(selected_candidates)
        except Exception as e:
            logger.exception("Oracle failed unexpectedly")
            msg = f"Cycle {self.cycle_count} failed at Oracle stage"
            raise RuntimeError(msg) from e

    def _step_refinement(self) -> Path:
        try:
            # Pass generator for memory safety
            structure_iterator = self._iterate_dataset()
            return self.trainer.train(structure_iterator, {}, self.workdir)
        except Exception as e:
            logger.exception("Trainer failed unexpectedly")
            msg = f"Cycle {self.cycle_count} failed at Trainer stage"
            raise RuntimeError(msg) from e

    def _step_validation(self, potential_path: Path) -> None:
        try:
            validation_result = self.validator.validate(potential_path)
            if not validation_result.passed:
                logger.warning(f"Validation failed: {validation_result.metrics}")
            else:
                logger.info("Validation passed.")
        except Exception as e:
            logger.exception("Validator failed unexpectedly")
            msg = f"Cycle {self.cycle_count} failed at Validator stage"
            raise RuntimeError(msg) from e

    def _step_deployment(self, potential_path: Path, start_struct: Structure) -> ExplorationResult:
        try:
            result = self.dynamics.run(potential_path, start_struct)
            logger.info(f"Dynamics finished with status: {result.status}")
        except Exception as e:
            logger.exception("Dynamics failed unexpectedly")
            msg = f"Cycle {self.cycle_count} failed at Dynamics stage"
            raise RuntimeError(msg) from e
        else:
            return result
