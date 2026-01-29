import logging

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.protocols import Phase

logger = logging.getLogger(__name__)

class TrainingPhase(Phase):
    @property
    def name(self) -> str:
        return "Training"

    def execute(self, state: WorkflowState, config: Config) -> WorkflowState:
        logger.info("Executing Training Phase...")

        work_dir = config.orchestrator.work_dir / f"cycle_{state.cycle_index:03d}" / "training"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Logic:
        # 1. Update dataset (pace_collect)
        # 2. Run training (pace_train)

        epochs = config.training.max_epochs
        logger.info(f"Training potential for {epochs} epochs...")

        # Stub: Update state with new potential path
        potential_path = work_dir / "potential.yace"
        state.current_potential_path = potential_path

        logger.info("Training Phase Completed.")
        return state
