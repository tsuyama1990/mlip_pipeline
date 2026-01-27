import json
import logging
import time

from dask.distributed import Client

from mlip_autopipec.config.models import MLIPConfig, SystemConfig
from mlip_autopipec.config.schemas.workflow import WorkflowConfig
from mlip_autopipec.data_models.state import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.dashboard import Dashboard, DashboardData
from mlip_autopipec.orchestration.database import DatabaseManager
from mlip_autopipec.orchestration.interfaces import BuilderProtocol, SurrogateProtocol
from mlip_autopipec.orchestration.phase_executor import PhaseExecutor
from mlip_autopipec.orchestration.task_queue import TaskQueue
from mlip_autopipec.utils.dask_utils import get_dask_client

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    The main orchestrator for the MLIP-AutoPipe workflow.

    Responsibilities:
    - Manage the high-level state machine (Cycle 06).
    - Persist state to disk.
    - Delegate execution to PhaseExecutor.
    """

    def __init__(
        self,
        config: SystemConfig | MLIPConfig,
        workflow_config: WorkflowConfig,
        builder: BuilderProtocol | None = None,
        surrogate: SurrogateProtocol | None = None,
    ) -> None:
        """
        Initialize the WorkflowManager.

        Args:
            config: The full SystemConfig or MLIPConfig.
            workflow_config: Configuration for the workflow (formerly orchestrator_config).
            builder: Optional dependency injection for Builder.
            surrogate: Optional dependency injection for Surrogate.
        """
        self.config = self._normalize_config(config)
        self.workflow_config = workflow_config
        self.work_dir = self.config.working_dir
        self.state_file = self.work_dir / "workflow_state.json"

        # Core Components
        self.db_manager = DatabaseManager(self.config.db_path)

        # Initialize Dask Client
        self.dask_client: Client = get_dask_client(
            scheduler_address=self.workflow_config.dask_scheduler_address
        )
        self.task_queue = TaskQueue(
            scheduler_address=self.workflow_config.dask_scheduler_address,
            workers=self.workflow_config.workers,
        )
        self.dashboard = Dashboard(self.work_dir, self.db_manager)

        # State
        self.state = self._load_state()

        # Dependencies
        self.builder = builder
        self.surrogate = surrogate

        # Phase Executor
        self.executor = PhaseExecutor(self)

    @staticmethod
    def _normalize_config(config: SystemConfig | MLIPConfig) -> SystemConfig:
        if isinstance(config, MLIPConfig):
            return SystemConfig(
                target_system=config.target_system,
                dft_config=config.dft,
                working_dir=config.runtime.work_dir,
                db_path=config.runtime.database_path,
                workflow_config=config.workflow_config,
                surrogate_config=config.surrogate_config,
                training_config=config.training_config,
                inference_config=config.inference_config,
                generator_config=config.generator_config,
            )
        if isinstance(config, SystemConfig):
            return config
        msg = "Invalid configuration type provided to WorkflowManager."
        raise TypeError(msg)

    def _load_state(self) -> WorkflowState:
        """Load or initialize workflow state."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                return WorkflowState.model_validate(data)
            except Exception:
                logger.exception("Failed to load state. Starting fresh.")
        return WorkflowState()

    def _save_state(self) -> None:
        """Persist state to disk."""
        self.state_file.write_text(self.state.model_dump_json(indent=2))

    def run(self) -> None:
        """
        Execute the main active learning loop.
        """
        logger.info("Starting Workflow Manager...")
        try:
            logger.debug(
                f"Entering run loop. Cycle: {self.state.cycle_index}, Max: {self.workflow_config.max_generations}"
            )
            while self.state.cycle_index < self.workflow_config.max_generations:
                logger.debug(f"Inside loop. Cycle: {self.state.cycle_index}")
                logger.info(
                    f"--- Cycle {self.state.cycle_index} | Phase: {self.state.current_phase.value} ---"
                )

                self._dispatch_phase()
                self._update_dashboard()
                self._save_state()

                time.sleep(1)

            logger.info("Max generations reached.")
        except Exception:
            logger.exception("Critical failure in run loop.")
        finally:
            self.task_queue.shutdown()
            if self.dask_client:
                self.dask_client.close()

    def _dispatch_phase(self) -> None:
        """Dispatch execution to the appropriate phase handler based on state."""
        phase = self.state.current_phase

        if phase == WorkflowPhase.EXPLORATION:
            # Run Inference/MD to explore and find uncertain structures
            # Returns True if halted (high uncertainty found), False otherwise
            logger.debug(f"Calling execute_inference on {self.executor}")
            halted = self.executor.execute_inference()
            logger.debug(f"execute_inference returned {halted}")
            if halted:
                logger.info(
                    "Exploration halted due to high uncertainty. Transitioning to Selection."
                )
                self.state.current_phase = WorkflowPhase.SELECTION
            else:
                logger.info("Exploration finished without halting. System Converged.")
                # Stop the loop by setting cycle_index to max
                self.state.cycle_index = self.workflow_config.max_generations

        elif phase == WorkflowPhase.SELECTION:
            self.executor.execute_selection()
            self.state.current_phase = WorkflowPhase.CALCULATION

        elif phase == WorkflowPhase.CALCULATION:
            self.executor.execute_dft()
            self.state.current_phase = WorkflowPhase.TRAINING

        elif phase == WorkflowPhase.TRAINING:
            self.executor.execute_training()
            self.state.current_phase = WorkflowPhase.EXPLORATION
            # Increment cycle index after training a new potential
            self.state.cycle_index += 1

    def _update_dashboard(self) -> None:
        """Update the dashboard."""
        try:
            count = self.db_manager.count()
        except Exception:
            count = 0

        # Using legacy DashboardData structure but mapping new fields
        data = DashboardData(
            generations=[self.state.cycle_index],
            structure_counts=[count],
            status=self.state.current_phase.value,
        )
        self.dashboard.update(data)
