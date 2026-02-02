import logging
from pathlib import Path

from mlip_autopipec.config.config_model import Config
from mlip_autopipec.orchestration.state import StateManager
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer


class Orchestrator:
    """
    The central coordinator of the Active Learning Cycle.
    Manages state, invokes physics engines, and handles the loop logic.
    """
    def __init__(self, config: Config, work_dir: Path | None = None) -> None:
        self.config = config
        self.work_dir = work_dir or Path.cwd()
        self.logger = logging.getLogger(__name__)

        # State persistence
        state_file = self.work_dir / "workflow_state.json"
        self.state_manager = StateManager(state_file)
        self.state = self.state_manager.load()

        self.trainer = PacemakerTrainer(config.training)

    def run(self) -> None:
        """Executes the Active Learning Cycle."""
        self.logger.info("Starting Orchestrator...")

        while self.state.iteration < self.config.orchestrator.max_iterations:
            current_iter = self.state.iteration
            self.logger.info(f"Starting Cycle {current_iter}")

            # Phase 1: Exploration (Mock for Cycle 01)
            self._run_exploration()

            # Phase 2: Selection & Calculation (Mock for Cycle 01)
            self._run_oracle()

            # Phase 3: Training
            self.logger.info("Phase 3: Training")
            potential_path = self.trainer.train(
                dataset=self.config.training.dataset_path,
                previous_potential=self.state.current_potential_path
            )

            # Update state
            self.state.update_potential(potential_path)
            self.state.increment_iteration()

            # Save state
            self.state_manager.save(self.state)
            self.logger.info(f"Cycle {current_iter} completed.")

        self.logger.info("Workflow completed successfully.")

    def _run_exploration(self) -> None:
        """
        Mock implementation of exploration phase.
        """
        self.logger.info("Phase 1: Exploration (Mock) - Skipping.")

    def _run_oracle(self) -> None:
        """
        Mock implementation of oracle phase.
        """
        self.logger.info("Phase 2: Oracle (Mock) - Skipping.")
