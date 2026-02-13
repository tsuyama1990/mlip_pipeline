import logging
from collections.abc import Iterator
from itertools import islice

from mlip_autopipec.core.active_learner import ActiveLearner
from mlip_autopipec.core.factory import ComponentFactory
from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.enums import TaskType
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import HaltInfo, Structure

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

        # Cycle 06: Active Learner
        self.active_learner = ActiveLearner(
            self.config, self.generator, self.oracle, self.trainer
        )

    def _run_cold_start_cycle(self, cycle: int) -> Potential:
        """
        Executes a Cold Start cycle: Generate -> Select -> Label -> Train.
        Used when no potential exists (typically Cycle 0).
        """
        logger.info(f"--- Cold Start (Cycle {cycle + 1}) ---")
        self.state_manager.update_cycle(cycle + 1)
        self.state_manager.save()

        # 1. Exploration
        self.state_manager.update_step(TaskType.EXPLORATION)
        self.state_manager.save()
        logger.info("Cold Start: Generating initial structures.")
        context = {"cycle": cycle, "temperature": self.config.dynamics.temperature}
        raw_candidates = self.generator.explore(context)

        # 2. Selection
        selected_candidates = self._select_cold_start(raw_candidates)

        # 3. Labeling (Oracle)
        self.state_manager.update_step(TaskType.TRAINING)
        self.state_manager.save()
        logger.info("Cold Start: Labeling structures (Oracle).")
        labeled_structures = list(self.oracle.compute(selected_candidates))

        if not labeled_structures:
            msg = "Cold start produced no labeled data."
            logger.error(msg)
            raise RuntimeError(msg)

        # 4. Training
        logger.info(f"Cold Start: Training on {len(labeled_structures)} structures.")
        new_potential = self.trainer.train(labeled_structures)

        # Update State
        self.state_manager.update_potential(new_potential.path)
        self.state_manager.save()
        logger.info(f"Cold Start: Potential trained at {new_potential.path}")

        # 5. Validation
        self.state_manager.update_step(TaskType.VALIDATION)
        self.state_manager.save()
        val_result = self.validator.validate(new_potential)
        if not val_result.passed:
            logger.warning("Cold Start: Validation failed.")

        return new_potential

    def _run_otf_cycle(self, cycle: int, potential: Potential) -> Potential:
        """
        Executes an OTF (On-The-Fly) Active Learning cycle.
        Runs dynamics, detects uncertainty, and triggers local learning loops.
        """
        logger.info(f"--- OTF Cycle {cycle + 1} ---")
        self.state_manager.update_cycle(cycle + 1)
        self.state_manager.save()

        # 1. Exploration (Dynamics)
        self.state_manager.update_step(TaskType.DYNAMICS)
        self.state_manager.save()

        logger.info("OTF: Starting Dynamics Loop.")
        seeds_iter = self.generator.explore({"cycle": cycle, "mode": "seed"})

        # Limit seeds
        max_seeds = self.config.orchestrator.max_otf_seeds
        halt_threshold = self.config.dynamics.max_gamma_threshold

        current_potential = potential

        for seed_idx, seed in enumerate(islice(seeds_iter, max_seeds)):
            if seed_idx > 0 and seed_idx % 10 == 0:
                logger.info(f"OTF Loop: Processed {seed_idx} seeds...")

            trajectory = self.dynamics.simulate(current_potential, seed)

            halted_frame = None
            for frame in trajectory:
                score = frame.uncertainty_score
                if score is not None and score > halt_threshold:
                    logger.info(f"Halt triggered: gamma={score:.2f}")
                    frame.provenance = "md_halt"
                    frame.metadata.update({
                        "halt_reason": "uncertainty_threshold",
                        "max_gamma": score,
                        "threshold": halt_threshold,
                    })
                    halted_frame = frame
                    break

            if halted_frame:
                logger.info("Halt event: Triggering Local Learning Loop...")

                # Active Learning Loop
                step = halted_frame.metadata.get("step", 0)
                if not isinstance(step, int):
                    step = 0

                halt_info = HaltInfo(
                    step=step,
                    max_gamma=halted_frame.uncertainty_score or 0.0,
                    structure=halted_frame,
                    reason="uncertainty_threshold"
                )

                # Update Potential
                new_potential = self.active_learner.process_halt(halt_info)

                # Validation Gate
                logger.info("OTF: Validating new potential...")
                val_result = self.validator.validate(new_potential)

                if val_result.passed:
                    logger.info("OTF: Validation passed. Updating potential.")
                    current_potential = new_potential
                    # Update State
                    self.state_manager.update_potential(new_potential.path)
                    self.state_manager.save()
                else:
                    logger.error("OTF: Validation failed! Discarding potential and aborting seed.")
                    # Abort this seed loop to prevent infinite cycling on bad potential
                    break

        logger.info("OTF: Dynamics Loop Completed.")
        return current_potential

    def _select_cold_start(self, candidates: Iterator[Structure]) -> Iterator[Structure]:
        """Applies random sampling and limits for Cold Start candidates."""
        max_samples = self.config.orchestrator.max_candidates
        ratio = self.config.trainer.selection_ratio

        if ratio <= 0:
            logger.warning("Selection ratio <= 0, defaulting to 1 (take all).")
            step = 1
        else:
            step = max(1, int(1.0 / ratio))

        stepped_iter = islice(candidates, 0, None, step)
        limited_iter = islice(stepped_iter, max_samples)

        yield from limited_iter

    def _execute_cycle(self, cycle: int, current_potential: Potential | None) -> Potential | None:
        """Executes a single cycle of the active learning workflow."""
        if current_potential is None:
            return self._run_cold_start_cycle(cycle)

        return self._run_otf_cycle(cycle, current_potential)

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
            try:
                current_potential = self._execute_cycle(cycle, current_potential)
            except Exception:
                logger.exception(f"Cycle {cycle + 1} failed")
                # We log and break to avoid corrupt state.
                break

        if self.config.orchestrator.cleanup_on_exit:
            self.state_manager.cleanup()

        logger.info("Orchestrator run completed.")
