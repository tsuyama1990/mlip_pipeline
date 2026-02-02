import logging
from pathlib import Path

from mlip_autopipec.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.orchestration.state import StateManager

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        config: Config,
        explorer: Explorer,
        oracle: Oracle,
        trainer: Trainer,
        validator: Validator | None = None,
    ) -> None:
        self.config = config
        self.explorer = explorer
        self.oracle = oracle
        self.trainer = trainer
        self.validator = validator

        self.state_manager = StateManager(Path("state.json"))

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

            cycle_work_dir = Path(f"cycle_{self.state.iteration}")
            cycle_work_dir.mkdir(exist_ok=True)

            # Phase 1: Exploration
            logger.info("Phase: Exploration")
            # In a real impl, we'd pass structures from exploration_result
            exploration_result = self.explorer.explore(
                self.state.current_potential_path, cycle_work_dir / "exploration"
            )

            # Phase 2: Selection & Calculation (Oracle)
            logger.info("Phase: Oracle")
            # In a real impl, we'd pass structures from exploration_result
            self.oracle.compute(exploration_result, cycle_work_dir / "oracle")

            # Phase 3: Training
            logger.info("Phase: Training")
            try:
                # In real impl, dataset path might come from oracle_result
                dataset_path = self.config.training.dataset_path

                potential_path = self.trainer.train(
                    dataset=dataset_path,
                    previous_potential=self.state.current_potential_path,
                )

                # Rename to unique filename to preserve history
                unique_path = Path(f"potential_iter_{self.state.iteration}.yace")
                if potential_path.exists():
                    potential_path.replace(unique_path)
                    potential_path = unique_path

                # Phase 4: Validation
                if self.validator:
                    logger.info("Phase: Validation")
                    self.validator.validate(unique_path)

                # Update state
                self.state.current_potential_path = unique_path
                self.state.history.append(
                    {
                        "iteration": self.state.iteration,
                        "potential": str(unique_path),
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
