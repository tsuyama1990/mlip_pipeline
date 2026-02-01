import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.domain_models.workflow import WorkflowState, WorkflowPhase
from mlip_autopipec.orchestration.phases.exploration import ExplorationPhase

logger = logging.getLogger("mlip_autopipec")


def run_one_shot(config: Config) -> JobResult:
    """
    Execute the One-Shot Pipeline (Cycle 02).
    Uses ExplorationPhase directly.
    """
    # Create dummy state for one-shot execution
    state = WorkflowState(
        project_name=config.project_name,
        dataset_path=Path("dummy_data.pckl"), # Not used in exploration
        current_phase=WorkflowPhase.EXPLORATION,
        generation=0
    )

    phase = ExplorationPhase()
    work_dir = Path(".")

    return phase.execute(state, config, work_dir)
