import logging
from collections.abc import Iterator
from itertools import islice

from mlip_autopipec.core.factory import ComponentFactory
from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.datastructures import Potential, Structure
from mlip_autopipec.domain_models.enums import TaskType

logger = logging.getLogger(__name__)


class Orchestrator:
    """The central controller for the Active Learning Pipeline."""

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.work_dir = config.orchestrator.work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.state_manager = StateManager(self.work_dir)
        self.factory = ComponentFactory(self.config)

        # Initialize components using factory
        self.generator = self.factory.create_generator()
        self.oracle = self.factory.create_oracle()
        self.trainer = self.factory.create_trainer(self.work_dir)
        self.dynamics = self.factory.create_dynamics(self.work_dir)
        self.validator = self.factory.create_validator()

    def _acquire_training_data(
        self, cycle: int, potential: Potential | None
    ) -> Iterator[Structure]:
        """
        Acquires training data either via Cold Start generation or OTF Active Learning.
        Returns an iterator of selected structures ready for labeling.
        """
        if potential is None:
            # Cold Start
            logger.info("Cold Start: Generating and selecting initial structures.")
            context = {"cycle": cycle, "temperature": self.config.dynamics.temperature}
            raw_candidates = self.generator.explore(context)
            return self._select_cold_start(raw_candidates)

        # OTF
        logger.info("OTF Mode: Running Active Learning Loop.")
        seeds = self.generator.explore({"cycle": cycle, "mode": "seed"})
        otf_candidates = self._otf_generator(seeds, potential)
        # Just limit the total number to avoid infinite loops or runaway costs
        return islice(otf_candidates, self.config.orchestrator.max_candidates)

    def _otf_generator(
        self, seeds_iter: Iterator[Structure], potential: Potential
    ) -> Iterator[Structure]:
        halt_threshold = self.config.dynamics.max_gamma_threshold
        n_local = self.config.generator.policy.n_local_candidates
        n_select = self.config.trainer.n_active_set_per_halt

        # Max seeds to process to prevent infinite loop if generator is infinite
        # Using a hard cap or config-based cap
        max_seeds_to_check = 1000

        # Stream seeds one by one to keep memory usage low
        # Use islice to enforce a hard limit on seed processing from infinite generators
        # This prevents unbounded loops if seeds_iter is infinite (e.g. RandomGenerator)
        safe_seeds_iter = islice(seeds_iter, max_seeds_to_check)

        for _seed_count, seed in enumerate(safe_seeds_iter):

            trajectory = self.dynamics.simulate(potential, seed)

            halted_frame = None
            for frame in trajectory:
                score = frame.uncertainty_score

                # Strict validation: Only halt if score is explicitly present and exceeds threshold
                if score is not None:
                    if score > halt_threshold:
                        logger.info(f"Halt triggered: gamma={score}")
                        frame.provenance = "md_halt"
                        frame.metadata.update(
                            {
                                "halt_reason": "uncertainty_threshold",
                                "max_gamma": score,
                                "threshold": halt_threshold,
                            }
                        )
                        halted_frame = frame
                        break
                else:
                    # Handle None case explicitly as requested by audit
                    logger.debug("Frame encountered with no uncertainty score during OTF.")
                    continue

            if halted_frame:
                logger.info("Halt event: Generating local candidates and selecting active set...")
                # 1. Generate local candidates
                candidates = self.generator.generate_local_candidates(halted_frame, count=n_local)

                # 2. Select active set (Local D-Optimality)
                # Delegated to Trainer's select_active_set which handles D-Optimality
                selected = self.trainer.select_active_set(candidates, count=n_select)

                yield from selected

    def _select_cold_start(self, candidates: Iterator[Structure]) -> Iterator[Structure]:
        """Applies random sampling and limits for Cold Start candidates."""
        max_samples = self.config.orchestrator.max_candidates
        ratio = self.config.trainer.selection_ratio
        step = max(1, int(1.0 / ratio)) if ratio > 0 else 1

        # Use itertools.islice to efficiently skip and limit items
        stepped_iter = islice(candidates, 0, None, step)
        limited_iter = islice(stepped_iter, max_samples)

        yield from limited_iter

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

            try:
                # 1. Exploration & Selection (Acquire Data)
                self.state_manager.update_step(TaskType.EXPLORATION)
                # This returns structures that are already selected/filtered
                structures_to_label = self._acquire_training_data(cycle, current_potential)

                # 2. Oracle
                self.state_manager.update_step(TaskType.TRAINING)
                labeled = self.oracle.compute(structures_to_label)

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
            except Exception:
                logger.exception(f"Cycle {cycle + 1} failed")
                # We might want to break or continue depending on severity.
                # For now, we log and break to avoid corrupt state.
                break

        if self.config.orchestrator.cleanup_on_exit:
            self.state_manager.cleanup()

        logger.info("Orchestrator run completed.")
