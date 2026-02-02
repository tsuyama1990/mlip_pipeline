import logging
from pathlib import Path

from mlip_autopipec.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.state import StateManager
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.state_manager = StateManager(Path("state.json"))

        self.trainer = PacemakerTrainer(config.training)

        # Initialize or load state
        loaded_state = self.state_manager.load()
        if loaded_state:
            self.state = loaded_state
            logger.info(f"Resumed from iteration {self.state.iteration}")
        else:
            self.state = WorkflowState()
            logger.info("Initialized new workflow state")

    def run(self) -> None:
        """Executes the Active Learning Cycle."""
        while self.state.iteration < self.config.orchestrator.max_iterations:
            logger.info(f"Starting Cycle {self.state.iteration}")

            # Phase 1: Exploration (Mock for Cycle 01)
            self._run_exploration()

            # Phase 2: Selection & Calculation (Mock for Cycle 01)
            self._run_oracle()

            # Phase 3: Training (Real implementation)
            logger.info("Starting Training Phase")
            try:
                potential_path = self.trainer.train(
                    dataset=self.config.training.dataset_path,
                    previous_potential=self.state.current_potential_path,
                )

                # Rename to unique filename to preserve history
                unique_path = Path(f"potential_iter_{self.state.iteration}.yace")
                if potential_path.exists():
                    potential_path.replace(unique_path)
                    potential_path = unique_path

                # Update state
                self.state.current_potential_path = potential_path
                self.state.history.append(
                    {
                        "iteration": self.state.iteration,
                        "potential": str(potential_path),
                        "status": "success",
                    }
                )

                # Increment iteration
                self.state.iteration += 1
                self.state_manager.save(self.state)
                logger.info(f"Cycle {self.state.iteration - 1} completed")

            except Exception:
                logger.exception(f"Cycle {self.state.iteration} failed")
                raise

    def _run_exploration(self) -> None:
        logger.info("Phase: Exploration (Mock)")

    def _run_oracle(self) -> None:
        logger.info("Phase: Oracle (Mock)")
