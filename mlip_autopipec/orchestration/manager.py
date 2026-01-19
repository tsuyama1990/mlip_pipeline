import json
import logging

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.orchestration.dashboard import Dashboard
from mlip_autopipec.orchestration.interfaces import BuilderProtocol, SurrogateProtocol
from mlip_autopipec.orchestration.models import DashboardData, OrchestratorConfig, WorkflowState
from mlip_autopipec.orchestration.phase_executor import PhaseExecutor
from mlip_autopipec.orchestration.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    The main orchestrator for the MLIP-AutoPipe workflow.

    Responsibilities:
    - Manage the high-level state machine (Generations and Phases).
    - Persist state to disk for resumption.
    - Delegate execution details to PhaseExecutor.
    """

    def __init__(
        self,
        config: SystemConfig,
        orchestrator_config: OrchestratorConfig,
        builder: BuilderProtocol | None = None,
        surrogate: SurrogateProtocol | None = None,
    ) -> None:
        """
        Initialize the WorkflowManager.

        Args:
            config: The full SystemConfig.
            orchestrator_config: Configuration for the orchestrator.
            builder: Optional dependency injection for Builder.
            surrogate: Optional dependency injection for Surrogate.
        """
        self.config = config
        self.orch_config = orchestrator_config
        self.work_dir = self.config.working_dir
        self.state_file = self.work_dir / "workflow_state.json"

        # Core Components
        self.db_manager = DatabaseManager(self.config.db_path)
        self.task_queue = TaskQueue(
            scheduler_address=self.orch_config.dask_scheduler_address,
            workers=self.orch_config.workers,
        )
        self.dashboard = Dashboard(self.work_dir, self.db_manager)

        # State
        self.state = self._load_state()

        # Dependencies
        self.builder = builder
        self.surrogate = surrogate

        # Phase Executor
        self.executor = PhaseExecutor(self)

    def _load_state(self) -> WorkflowState:
        """Load or initialize workflow state."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                return WorkflowState(**data)
            except Exception:
                logger.exception("Failed to load state. Starting fresh.")
        return WorkflowState(current_generation=0, status="idle")

    def _save_state(self) -> None:
        """Persist state to disk."""
        self.state_file.write_text(self.state.model_dump_json(indent=2))

    def run(self) -> None:
        """
        Execute the main loop.
        """
        logger.info("Starting Workflow Manager...")
        try:
            while self.state.current_generation < self.orch_config.max_generations:
                logger.info(f"--- Generation {self.state.current_generation} ---")

                self._dispatch_phase()
                self._update_dashboard()
                self._save_state()

            logger.info("Max generations reached.")
        except Exception:
            logger.exception("Critical failure in run loop.")
        finally:
            self.task_queue.shutdown()

    def _dispatch_phase(self) -> None:
        """Dispatch execution to the appropriate phase handler based on state."""
        if self.state.status == "idle":
            self.executor.execute_exploration()
            self.state.status = "dft"

        elif self.state.status == "dft":
            self.executor.execute_dft()
            self.state.status = "training"

        elif self.state.status == "training":
            self.executor.execute_training()
            self.state.status = "inference"

        elif self.state.status == "inference":
            self.executor.execute_inference()
            self.state.current_generation += 1
            self.state.status = "idle"

    def _update_dashboard(self) -> None:
        """Update the dashboard."""
        try:
            count = self.db_manager.count()
        except Exception:
            count = 0

        data = DashboardData(
            generations=[self.state.current_generation],
            structure_counts=[count],
            status=self.state.status,
        )
        self.dashboard.update(data)
