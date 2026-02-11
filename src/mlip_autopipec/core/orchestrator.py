import logging

from mlip_autopipec.constants import DEFAULT_LOG_FILE, DEFAULT_STATE_FILE
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models import GlobalConfig, WorkflowStatus

logger = logging.getLogger("mlip_autopipec")


class Orchestrator:
    """
    The main control loop for the active learning workflow.
    """

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.work_dir = config.orchestrator.work_dir
        self.max_cycles = config.orchestrator.max_cycles

        # Ensure work dir exists
        if self.work_dir.exists() and not self.work_dir.is_dir():
            msg = f"{self.work_dir} is not a directory."
            raise NotADirectoryError(msg)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Setup Logging
        setup_logging(self.work_dir / DEFAULT_LOG_FILE)

        # State Manager
        self.state_manager = StateManager(self.work_dir / DEFAULT_STATE_FILE)
        self.state = self.state_manager.load()

        logger.info(f"Orchestrator initialized in {self.work_dir}")
        if self.state.iteration > 0:
            logger.info(f"Resuming from iteration {self.state.iteration}")

    def run(self) -> None:
        """
        Executes the main active learning loop.
        """
        logger.info(f"Starting Orchestrator with max_cycles={self.max_cycles}")

        while self.state.iteration < self.max_cycles:
            current_iter = self.state.iteration + 1
            logger.info(f"Starting cycle {current_iter}...")

            # 1. Exploration
            self.state.status = WorkflowStatus.EXPLORATION
            self.state_manager.save(self.state)
            self.explore()

            # 2. Labeling
            self.state.status = WorkflowStatus.LABELING
            self.state_manager.save(self.state)
            self.label()

            # 3. Training
            self.state.status = WorkflowStatus.TRAINING
            self.state_manager.save(self.state)
            self.train()

            # 4. Update Iteration
            self.state.iteration += 1
            self.state_manager.save(self.state)
            logger.info(f"Cycle {current_iter} completed.")

        self.state.status = WorkflowStatus.COMPLETED
        self.state_manager.save(self.state)
        logger.info("Workflow completed successfully.")

    def explore(self) -> None:
        logger.info("Exploring configuration space (Mock)...")

    def label(self) -> None:
        logger.info("Labeling structures with Oracle (Mock)...")

    def train(self) -> None:
        logger.info("Training potential (Mock)...")
