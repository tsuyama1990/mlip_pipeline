import logging
from pathlib import Path

from mlip_autopipec.config.models import UserInputConfig
from mlip_autopipec.domain_models.state import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.orchestration.phases.dft import DFTPhase
from mlip_autopipec.orchestration.phases.exploration import ExplorationPhase
from mlip_autopipec.orchestration.phases.selection import SelectionPhase
from mlip_autopipec.orchestration.phases.training import TrainingPhase
from mlip_autopipec.orchestration.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Orchestrates the active learning cycles.
    """

    def __init__(self, config: UserInputConfig, work_dir: Path, state_file: Path | None = None, workflow_config=None):
        self.config = config
        self.work_dir = work_dir
        self.state_file = state_file or (work_dir / "state.json")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Allow workflow_config override or use from config
        self.workflow_config = workflow_config or config.workflow_config

        # Use config value for db path, not hardcoded
        self.db_path = self.work_dir / self.config.runtime.database_path
        self.db_manager = DatabaseManager(self.db_path)

        self.state = self._load_state()

        # Initialize TaskQueue
        workers = self.workflow_config.workers if self.workflow_config else 4
        self.task_queue = TaskQueue(workers=workers)

    def _load_state(self) -> WorkflowState:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return WorkflowState.model_validate_json(f.read())
        return WorkflowState()

    def save_state(self) -> None:
        with open(self.state_file, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def run(self) -> None:
        """Main execution loop for full active learning (Future Cycles)."""
        max_cycles = self.workflow_config.max_generations if self.workflow_config else 5

        logger.info(f"Starting Workflow. Max Cycles: {max_cycles}")

        try:
            while self.state.cycle_index < max_cycles:
                self.run_cycle()
                self.state.cycle_index += 1
                self.save_state()

            logger.info("Workflow Completed.")

        except Exception:
            logger.exception("Workflow Interrupted.")
            raise
        finally:
            self.task_queue.shutdown()

    def run_cycle(self) -> None:
        """Executes a single Active Learning Cycle."""
        cycle = self.state.cycle_index
        logger.info(f"=== Starting Cycle {cycle} ===")

        # Phase A: Exploration
        self.state.current_phase = WorkflowPhase.EXPLORATION
        self.save_state()
        ExplorationPhase(self).execute()

        # Phase B: Selection
        # Only run selection if we have a potential to select with (Active Learning)
        if cycle > 0 and self.state.latest_potential_path:
            self.state.current_phase = WorkflowPhase.SELECTION
            self.save_state()
            SelectionPhase(self).execute()

        # Phase C: Calculation (DFT)
        self.state.current_phase = WorkflowPhase.CALCULATION
        self.save_state()
        DFTPhase(self).execute()

        # Phase D: Training
        self.state.current_phase = WorkflowPhase.TRAINING
        self.save_state()
        TrainingPhase(self).execute()

        logger.info(f"=== Cycle {cycle} Completed ===")
