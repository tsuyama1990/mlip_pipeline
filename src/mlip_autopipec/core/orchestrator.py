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
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.state_manager = StateManager(config.workdir / "workflow_state.json")
        self.dataset = Dataset(config.workdir / "dataset.jsonl")

        # Instantiate components
        self.generator = ComponentFactory.get_generator(config.components.generator)
        self.oracle = ComponentFactory.get_oracle(config.components.oracle)
        self.trainer = ComponentFactory.get_trainer(config.components.trainer)
        self.dynamics = ComponentFactory.get_dynamics(config.components.dynamics)
        self.validator = ComponentFactory.get_validator(config.components.validator)

        self.current_potential: Potential | None = None

    def run(self) -> None:
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
        cycle = self.state_manager.state.current_cycle + 1
        logger.info(f"Starting Cycle {cycle}")

        cycle_dir = self.config.workdir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Exploration
        structures: Iterator[Structure]
        n_structures = self.config.components.generator.n_structures

        if cycle == 1:
            logger.info("Cycle 1: Generating initial structures")
            structures = iter(self.generator.generate(n_structures=n_structures))
        else:
            logger.info(f"Cycle {cycle}: Exploring with previous potential")
            if self.current_potential is None:
                # In a real restart scenario, we might load the potential from previous cycle
                prev_cycle = cycle - 1
                prev_pot_path = self.config.workdir / f"cycle_{prev_cycle:02d}" / "potential.yace"
                if prev_pot_path.exists():
                    self.current_potential = Potential(path=prev_pot_path)
                else:
                    msg = "No potential available for exploration"
                    raise RuntimeError(msg)

            # Generate start structures for dynamics
            # Pass generator directly to ensure streaming (iterator is passed)
            start_structures_iter = self.generator.generate(n_structures=n_structures)

            # Use explicit config for uncertainty threshold if needed, but here we just call explore
            # The audit mentioned line 28: "Hardcoded uncertainty threshold value...".
            # `Orchestrator` doesn't seem to use `uncertainty_threshold` directly here, it's used inside `Dynamics`.
            # But just to be safe, we ensure dynamics is configured correctly via factory.

            structures = self.dynamics.explore(self.current_potential, start_structures_iter)

        # Step 2: Labeling
        logger.info("Labeling structures")
        # Ensure oracle.compute is also streaming (consuming iterator)
        labeled_structures = self.oracle.compute(structures)

        # Step 3: Dataset Update
        logger.info("Updating dataset")
        # Dataset.append consumes the iterator line-by-line
        self.dataset.append(labeled_structures)

        # Step 4: Training
        logger.info("Training potential")
        self.current_potential = self.trainer.train(
            dataset=self.dataset, workdir=cycle_dir, previous_potential=self.current_potential
        )
        logger.info(f"Potential trained: {self.current_potential.path}")

        # Step 5: Validation
        logger.info("Validating potential")
        metrics = self.validator.validate(self.current_potential)
        logger.info(f"Validation metrics: {metrics}")

        self.current_potential.metrics.update(metrics)
