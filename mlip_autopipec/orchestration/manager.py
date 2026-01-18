import json
import logging

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.orchestration.dashboard import Dashboard
from mlip_autopipec.orchestration.models import DashboardData, OrchestratorConfig, WorkflowState
from mlip_autopipec.orchestration.task_queue import TaskQueue

# Import modules (Using mocks or interfaces if not available yet)
# We assume these are available based on previous cycles
# from mlip_autopipec.generator import Generator # Cycle 03
# from mlip_autopipec.surrogate import SurrogateClient # Cycle 04
# from mlip_autopipec.dft.runner import QERunner # Cycle 02
# from mlip_autopipec.training.pacemaker import PacemakerWrapper # Cycle 05
# from mlip_autopipec.inference.runner import LammpsRunner # Cycle 06

logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    The main orchestrator for the MLIP-AutoPipe workflow.
    Managed the state machine and transitions between phases.
    """
    def __init__(self, config: SystemConfig, orchestrator_config: OrchestratorConfig):
        """
        Initialize the WorkflowManager.

        Args:
            config: The full SystemConfig.
            orchestrator_config: Configuration for the orchestrator.
        """
        self.config = config
        self.orch_config = orchestrator_config
        self.work_dir = self.config.working_dir
        self.state_file = self.work_dir / "workflow_state.json"

        # Initialize Core Components
        self.db_manager = DatabaseManager(self.config.db_path)
        self.task_queue = TaskQueue(
            scheduler_address=self.orch_config.dask_scheduler_address,
            workers=self.orch_config.workers
        )
        self.dashboard = Dashboard(self.work_dir, self.db_manager)

        # Load or Initialize State
        self.state = self._load_state()

    def _load_state(self) -> WorkflowState:
        """
        Load state from disk or create a new one.
        """
        if self.state_file.exists():
            logger.info(f"Loading existing workflow state from {self.state_file}")
            try:
                data = json.loads(self.state_file.read_text())
                return WorkflowState(**data)
            except Exception as e:
                logger.error(f"Failed to load state file: {e}. Starting fresh.")
                return WorkflowState(current_generation=0, status="idle")
        else:
            logger.info("No existing state found. Starting fresh.")
            return WorkflowState(current_generation=0, status="idle")

    def _save_state(self) -> None:
        """
        Save current state to disk.
        """
        logger.info(f"Saving workflow state to {self.state_file}")
        self.state_file.write_text(self.state.model_dump_json(indent=2))

    def run(self) -> None:
        """
        Execute the main workflow loop.
        """
        logger.info("Starting Workflow Manager...")

        while self.state.current_generation < self.orch_config.max_generations:
            logger.info(f"--- Generation {self.state.current_generation} ---")

            # Phase A: Exploration / Generation
            if self.state.status == "idle":
                 self._run_exploration_phase()

            # Phase B: DFT Labeling
            if self.state.status == "dft":
                self._run_dft_phase()

            # Phase C: Training
            if self.state.status == "training":
                self._run_training_phase()

            # Phase D: Inference / Exploitation
            if self.state.status == "inference":
                self._run_inference_phase()

            # End of loop updates
            self._update_dashboard()
            self._save_state()

        logger.info("Max generations reached. Workflow completed.")
        self.task_queue.shutdown()

    def _run_exploration_phase(self) -> None:
        """
        Phase A: Generate structures (SQS, NMS) and Pre-screen with Surrogate.
        """
        logger.info("Phase A: Exploration")
        # TODO: Implement actual logic:
        # 1. Check if DB has enough data.
        # 2. If not, call Generator -> Surrogate -> FPS -> Candidates.
        # 3. Add Candidates to pending_tasks.

        # For now, we simulate transitions
        self.state.status = "dft"
        self._save_state()

    def _run_dft_phase(self) -> None:
        """
        Phase B: Run DFT on pending structures.
        """
        logger.info("Phase B: DFT Labeling")

        # TODO: Implement actual logic:
        # 1. Retrieve pending candidates.
        # 2. Submit to TaskQueue.
        # 3. Wait for results.
        # 4. Ingest results to DB.

        self.state.status = "training"
        self._save_state()

    def _run_training_phase(self) -> None:
        """
        Phase C: Train the potential.
        """
        logger.info("Phase C: Training")

        # TODO: Implement actual logic:
        # 1. Export Dataset.
        # 2. Run Pacemaker.
        # 3. Validate potential.

        self.state.status = "inference"
        self._save_state()

    def _run_inference_phase(self) -> None:
        """
        Phase D: Run MD and Active Learning.
        """
        logger.info("Phase D: Inference")

        # TODO: Implement actual logic:
        # 1. Run LAMMPS with new potential.
        # 2. Check for uncertainty.
        # 3. If high uncertainty, extract structures -> Add to pending -> Loop.
        # 4. Else, increment generation.

        self.state.current_generation += 1
        self.state.status = "idle" # Loop back
        self._save_state()

    def _update_dashboard(self) -> None:
        """
        Collect stats and update dashboard.
        """
        # Get real counts from DB
        try:
            total_structures = self.db_manager.count()
        except Exception:
            total_structures = 0

        # We keep the history accumulation logic simple for now by just checking current state.
        # In a real scenario, we'd query history from DB metadata or logs.
        # For now, we construct a data point for the current generation.

        # Note: Dashboard expects lists for history.
        # Since we don't persist history in WorkflowState, we construct a snapshot.
        # In a real system, we'd load the full history.

        data = DashboardData(
            generations=[self.state.current_generation],
            rmse_values=[0.0],  # Placeholder until we track metrics
            structure_counts=[total_structures],
            status=self.state.status
        )
        self.dashboard.update(data)
