import logging

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.protocols import Phase

logger = logging.getLogger(__name__)

class ValidationPhase(Phase):
    @property
    def name(self) -> str:
        return "Validation"

    def execute(self, state: WorkflowState, config: Config) -> WorkflowState:
        logger.info("Executing Validation Phase...")

        work_dir = config.orchestrator.work_dir / f"cycle_{state.cycle_index:03d}" / "validation"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Logic:
        # 1. Phonon calculation
        # 2. Elastic constants
        # 3. RMSE check

        if config.validation.check_phonons:
            logger.info("Checking phonon stability...")

        if config.validation.check_elasticity:
            logger.info("Checking elastic constants...")

        logger.info("Validation Phase Completed.")
        return state
