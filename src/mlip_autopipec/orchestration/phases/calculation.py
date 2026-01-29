import logging

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.protocols import Phase

logger = logging.getLogger(__name__)

class CalculationPhase(Phase):
    @property
    def name(self) -> str:
        return "Calculation"

    def execute(self, state: WorkflowState, config: Config) -> WorkflowState:
        logger.info("Executing Calculation Phase (DFT)...")

        work_dir = config.orchestrator.work_dir / f"cycle_{state.cycle_index:03d}" / "dft"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Logic:
        # 1. Iterate over selected candidates
        # 2. Run DFT (Quantum Espresso / VASP)
        # 3. Store results in Candidate objects

        logger.info(f"Running DFT using command: {config.dft.command}")

        # Stub: Assume success

        logger.info("Calculation Phase Completed.")
        return state
