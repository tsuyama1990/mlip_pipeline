import logging

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.protocols import Phase

logger = logging.getLogger(__name__)

class SelectionPhase(Phase):
    @property
    def name(self) -> str:
        return "Selection"

    def execute(self, state: WorkflowState, config: Config) -> WorkflowState:
        logger.info("Executing Selection Phase...")

        work_dir = config.orchestrator.work_dir / f"cycle_{state.cycle_index:03d}" / "selection"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Logic:
        # 1. Load halted structures or trajectory
        # 2. Filter by uncertainty (config.selection.uncertainty_threshold)
        # 3. Select active set

        n_candidates = config.selection.n_candidates
        logger.info(f"Selecting top {n_candidates} candidates based on uncertainty.")

        # Stub: No candidates actually selected in this architectural skeleton

        logger.info("Selection Phase Completed.")
        return state
