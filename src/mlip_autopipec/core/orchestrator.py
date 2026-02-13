import logging
from collections.abc import Iterator

from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.datastructures import Potential, Structure
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TaskType,
    TrainerType,
    ValidatorType,
)
from mlip_autopipec.dynamics import BaseDynamics, MockDynamics
from mlip_autopipec.generator import (
    AdaptiveGenerator,
    BaseGenerator,
    M3GNetGenerator,
    MockGenerator,
    RandomGenerator,
)
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
        gen_type = self.config.generator.type
        if gen_type == GeneratorType.MOCK:
            return MockGenerator()
        if gen_type == GeneratorType.RANDOM:
            return RandomGenerator(self.config.generator)
        if gen_type == GeneratorType.M3GNET:
            return M3GNetGenerator(self.config.generator)
        if gen_type == GeneratorType.ADAPTIVE:
            return AdaptiveGenerator(self.config.generator)

        msg = f"Unsupported generator type: {gen_type}"
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
            return MockDynamics(self.config.dynamics)
        msg = f"Unsupported dynamics type: {self.config.dynamics.type}"
        raise ValueError(msg)

    def _create_validator(self) -> BaseValidator:
        if self.config.validator.type == ValidatorType.MOCK:
            return MockValidator()
        msg = f"Unsupported validator type: {self.config.validator.type}"
        raise ValueError(msg)

    def _explore_candidates(self, cycle: int, potential: Potential | None) -> Iterator[Structure]:
        if potential is None:
            logger.info("Cold Start: Using Generator for initial structures.")
            context = {"cycle": cycle, "temperature": self.config.dynamics.temperature}
            return self.generator.explore(context)

        logger.info("OTF Mode: Using Dynamics with active potential.")
        seeds = self.generator.explore({"cycle": cycle, "mode": "seed"})
        return self._otf_generator(seeds, potential)

    def _otf_generator(self, seeds_iter: Iterator[Structure], potential: Potential) -> Iterator[Structure]:
        halt_threshold = self.config.dynamics.max_gamma_threshold
        n_local = self.config.generator.policy.n_local_candidates
        n_select = self.config.trainer.n_active_set_per_halt

        for seed in seeds_iter:
            trajectory = self.dynamics.simulate(potential, seed)

            halted_frame = None
            for frame in trajectory:
                score = frame.uncertainty_score

                if score is not None and score > halt_threshold:
                    logger.info(f"Halt triggered: gamma={score}")
                    frame.provenance = "md_halt"
                    halted_frame = frame
                    break
                # If score is None or low, continue simulation

            if halted_frame:
                logger.info("Halt event: Generating local candidates and selecting active set...")
                # 1. Generate local candidates
                candidates = self.generator.generate_local_candidates(halted_frame, count=n_local)

                # 2. Select active set (Local D-Optimality)
                selected = self.trainer.select_active_set(candidates, count=n_select)

                # Streaming check - if iterator is empty, nothing happens, which is safe.
                # However, for debugging, we might want to know if selection failed.
                yield from selected

    def _select_candidates(self, candidates: Iterator[Structure]) -> Iterator[Structure]:
        max_samples = self.config.orchestrator.max_candidates
        # Simple sampling based on selection_ratio (streaming approximation)
        ratio = self.config.trainer.selection_ratio
        step = max(1, int(1.0 / ratio)) if ratio > 0 else 1

        accepted = 0
        for i, s in enumerate(candidates):
            if i % step == 0:
                yield s
                accepted += 1
                if accepted >= max_samples:
                    logger.info(f"Selection limit reached ({max_samples}).")
                    break

    def run(self) -> None:
        """Executes the active learning workflow."""
        logger.info("Starting Orchestrator...")

        # Load state
        start_cycle = self.state_manager.state.current_cycle
        max_cycles = self.config.orchestrator.max_cycles

        # Load potential
        current_potential = None
        pot_path = self.state_manager.state.active_potential_path
        if pot_path and pot_path.exists():
            current_potential = Potential(path=pot_path, format="yace")

        for cycle in range(start_cycle, max_cycles):
            self.state_manager.update_cycle(cycle + 1)
            logger.info(f"--- Starting Cycle {cycle + 1}/{max_cycles} ---")

            # 1. Exploration
            self.state_manager.update_step(TaskType.EXPLORATION)
            candidates = self._explore_candidates(cycle, current_potential)

            # 2. Selection
            selected = self._select_candidates(candidates)

            # 3. Oracle
            self.state_manager.update_step(TaskType.TRAINING)
            labeled = self.oracle.compute(selected)

            # 4. Training
            self.state_manager.update_step(TaskType.TRAINING)
            new_potential = self.trainer.train(labeled)
            self.state_manager.update_potential(new_potential.path)
            current_potential = new_potential
            self.state_manager.save()
            logger.info(f"Potential trained: {new_potential.path}")

            # 5. Validation
            self.state_manager.update_step(TaskType.VALIDATION)
            val_result = self.validator.validate(new_potential)
            if not val_result.passed:
                logger.warning("Validation FAILED.")

        if self.config.orchestrator.cleanup_on_exit:
            self.state_manager.cleanup()

        logger.info("Orchestrator run completed.")
