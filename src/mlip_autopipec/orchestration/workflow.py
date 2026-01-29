import logging
from pathlib import Path

from mlip_autopipec.constants import DEFAULT_STATE_FILENAME
from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.infrastructure import io

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages the active learning workflow state and execution.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = WorkflowState()
        self.state_path = Path(DEFAULT_STATE_FILENAME)

    def load_state(self) -> None:
        """Load the workflow state from disk if it exists."""
        if self.state_path.exists():
            self.state = io.load_state(self.state_path)
            logger.info("Loaded existing workflow state from %s", self.state_path)
        else:
            logger.info("No existing state found. Starting fresh.")
            self.state = WorkflowState()

    def save_state(self) -> None:
        """Save the current workflow state to disk."""
        io.save_state(self.state, self.state_path)
        logger.debug("Saved workflow state to %s", self.state_path)

    def run(self) -> None:
        """
        Execute the active learning loop.
        For Cycle 01, this simply initializes the state and logs the start.
        """
        logger.info("Starting MLIP Active Learning Loop")

        self.load_state()

        # Ensure state is valid for running
        self.state.is_halted = False
        if self.state.current_phase == WorkflowPhase.INITIALIZATION:
            # Transition to next phase would happen here in future cycles
            pass

        logger.info("Workflow initialized.")

        self.save_state()
