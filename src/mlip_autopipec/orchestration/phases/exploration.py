import logging

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.protocols import Phase

logger = logging.getLogger(__name__)

class ExplorationPhase(Phase):
    @property
    def name(self) -> str:
        return "Exploration"

    def execute(self, state: WorkflowState, config: Config) -> WorkflowState:
        logger.info("Executing Exploration Phase...")

        work_dir = config.orchestrator.work_dir / f"cycle_{state.cycle_index:03d}" / "exploration"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Logic to decide if we run MD or use Generator
        # For now, simulate MD run
        logger.info(f"Running MD exploration with temp={config.exploration.temperature}K")

        # Simulate result
        # In a real scenario, this would check log files for 'halt'
        # Let's assume successful completion for this stub, or random halt?
        # Let's keep it simple: No halt.
        state.is_halted = False

        logger.info("Exploration Phase Completed.")
        return state
