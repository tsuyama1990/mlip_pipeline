"""Workflow manager for orchestrating the active learning loop."""

import logging
from pathlib import Path

from mlip_autopipec.constants import DEFAULT_STATE_FILENAME
from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.infrastructure import io

logger = logging.getLogger(__name__)


class WorkflowManager:
    """Manages the lifecycle of the active learning workflow."""

    def __init__(self, config: Config, state_path: Path = Path(DEFAULT_STATE_FILENAME)) -> None:
        self.config = config
        self.state_path = state_path
        self.state: WorkflowState = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> WorkflowState:
        """Load state from disk or initialize a new one."""
        if self.state_path.exists():
            try:
                data = io.load_json(self.state_path)
                logger.info(f"Loaded existing workflow state from {self.state_path}")
                return WorkflowState(**data)
            except Exception:
                logger.exception("Failed to load state. Starting fresh.")
                # In a real scenario, we might want to backup the corrupted state

        logger.info("Initializing new workflow state.")
        return WorkflowState(
            cycle_index=0,
            current_phase=WorkflowPhase.INITIALIZATION
        )

    def save_state(self) -> None:
        """Persist current state to disk."""
        try:
            io.save_json(self.state.model_dump(mode="json"), self.state_path)
            logger.debug(f"Saved workflow state to {self.state_path}")
        except Exception:
            logger.exception("Failed to save state.")

    def run(self) -> None:
        """Execute the workflow loop."""
        logger.info("Starting MLIP Active Learning Loop")

        # Cycle 01: Stub implementation
        # Just save the state to prove it works
        self.state.current_phase = WorkflowPhase.EXPLORATION
        self.save_state()

        logger.info("Workflow initialized (Stub)")
