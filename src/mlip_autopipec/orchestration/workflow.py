"""Workflow manager for orchestrating the active learning loop."""

import logging
from pathlib import Path

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.state_manager import StateManager

logger = logging.getLogger(__name__)


class WorkflowManager:
    """Manages the lifecycle of the active learning workflow."""

    def __init__(self, config: Config, state_path: Path | None = None) -> None:
        self.config = config
        # Use config state_file if not provided override
        path = state_path if state_path else config.orchestrator.state_file

        self.state_manager = StateManager(path)
        self.state: WorkflowState = self.state_manager.load_or_initialize()

    def run(self) -> None:
        """Execute the workflow loop."""
        logger.info("Starting MLIP Active Learning Loop")

        # Cycle 01: Stub implementation
        # Just save the state to prove it works
        self.state.current_phase = WorkflowPhase.EXPLORATION
        self.state_manager.save(self.state)

        logger.info("Workflow initialized (Stub)")
