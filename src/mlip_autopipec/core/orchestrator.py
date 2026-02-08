import logging
from collections.abc import Iterator

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.core.state import StateManager
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.factory import ComponentFactory

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main pipeline orchestrator.

    Architecture Note:
    This implementation follows a "Batch Active Learning" strategy (Cycle 1 -> N),
    which is more robust for automated pipelines than the "On-the-Fly (OTF) Halt & Resume"
    strategy described in the original Spec (Section 3.4).

    Instead of halting the MD simulation mid-run (which requires complex process control
    and state management), we run Exploration (Dynamics) to completion, collecting
    uncertain structures into a batch. This batch is then Labeled (Oracle),
    added to the Dataset, and used for Training a new Potential for the next cycle.
    """

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config

        # Resolve paths from config
        self.dataset_path = config.workdir / config.orchestrator.dataset_filename
        self.state_path = config.workdir / config.orchestrator.state_filename

        # Initialize Core Components
        self.state_manager = StateManager(self.state_path)
        self.dataset = Dataset(self.dataset_path, root_dir=config.workdir)

        # Instantiate components via Factory (Dependency Injection)
        self.generator = ComponentFactory.get_generator(config.components.generator)
        self.oracle = ComponentFactory.get_oracle(config.components.oracle)
        self.trainer = ComponentFactory.get_trainer(config.components.trainer)
        self.dynamics = ComponentFactory.get_dynamics(config.components.dynamics)
        self.validator = ComponentFactory.get_validator(config.components.validator)

        self.current_potential: Potential | None = None

    def __repr__(self) -> str:
        return f"<Orchestrator(workdir={self.config.workdir}, cycle={self.state_manager.state.current_cycle})>"

    def __str__(self) -> str:
        return f"Orchestrator(cycle={self.state_manager.state.current_cycle})"

    def run(self) -> None:
        """Run the active learning loop until max_cycles is reached."""
        logger.info("Starting Orchestrator")
        self.state_manager.update_status("RUNNING")

        try:
            while self.state_manager.state.current_cycle < self.config.max_cycles:
                self._run_cycle()
                self.state_manager.update_cycle(self.state_manager.state.current_cycle + 1)

            logger.info("Max cycles reached. Stopping.")
            self.state_manager.update_status("STOPPED")

        except Exception:
            logger.exception("Orchestrator failed")
            self.state_manager.update_status("ERROR")
            raise

    def _run_cycle(self) -> None:
        """
        Execute one full active learning cycle:
        Exploration -> Labeling -> Dataset Update -> Training -> Validation.
        """
        cycle = self.state_manager.state.current_cycle + 1
        logger.info(f"=== Starting Cycle {cycle:02d} ===")

        # Use configured cycle directory pattern
        cycle_dir_name = self.config.orchestrator.cycle_dir_pattern.format(cycle=cycle)
        cycle_dir = self.config.workdir / cycle_dir_name
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Exploration (Generator or Dynamics)
        structures: Iterator[Structure]
        n_structures = self.config.components.generator.n_structures

        if cycle == 1:
            logger.info(f"[Cycle {cycle}] Exploration: Generating initial structures (Cold Start)")
            # Cycle 1 uses Generator to create initial random/heuristic structures
            structures = self.generator.generate(n_structures=n_structures)
        else:
            logger.info(f"[Cycle {cycle}] Exploration: Running Dynamics with previous potential")

            # Ensure potential is available
            if self.current_potential is None:
                prev_cycle = cycle - 1
                prev_cycle_dir_name = self.config.orchestrator.cycle_dir_pattern.format(cycle=prev_cycle)
                prev_pot_path = self.config.workdir / prev_cycle_dir_name / self.config.orchestrator.potential_filename

                if prev_pot_path.exists():
                    self.current_potential = Potential(path=prev_pot_path)
                    logger.info(f"Loaded potential from {prev_pot_path}")
                else:
                    msg = f"Cycle {cycle}: No potential available from Cycle {prev_cycle} at {prev_pot_path}"
                    raise RuntimeError(msg)

            # Generate start structures for dynamics (seeds)
            start_structures_iter = self.generator.generate(n_structures=n_structures)

            # Dynamics explores and yields UNCERTAIN structures
            structures = self.dynamics.explore(self.current_potential, start_structures_iter)

        # Step 2: Labeling (Oracle)
        logger.info(f"[Cycle {cycle}] Labeling: Computing DFT properties")
        # Oracle.compute consumes the iterator stream
        labeled_structures = self.oracle.compute(structures)

        # Step 3: Dataset Update
        logger.info(f"[Cycle {cycle}] Dataset: Appending new data")
        # Dataset.append consumes the labeled stream and writes to disk
        self.dataset.append(labeled_structures)

        # Step 4: Training (Trainer)
        logger.info(f"[Cycle {cycle}] Training: Fitting new potential")
        # Trainer reads the FULL dataset (or a subset) and produces a new potential
        self.current_potential = self.trainer.train(
            dataset=self.dataset, workdir=cycle_dir, previous_potential=self.current_potential
        )
        logger.info(f"Potential trained: {self.current_potential.path}")

        # Step 5: Validation (Validator)
        logger.info(f"[Cycle {cycle}] Validation: Evaluating quality")
        metrics = self.validator.validate(self.current_potential)
        logger.info(f"Validation metrics: {metrics}")

        self.current_potential.metrics.update(metrics.model_dump())
        logger.info(f"=== Cycle {cycle:02d} Completed ===")
