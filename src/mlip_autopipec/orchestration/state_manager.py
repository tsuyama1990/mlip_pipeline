"""State manager for handling workflow persistence."""

import logging
import shutil
import time
from pathlib import Path

from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.infrastructure import io

logger = logging.getLogger(__name__)


class StateManager:
    """Handles loading and saving of workflow state."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def load_or_initialize(self) -> WorkflowState:
        """Load state from disk or initialize a new one."""
        if self.state_path.exists():
            try:
                data = io.load_json(self.state_path)
                logger.info(f"Loaded existing workflow state from {self.state_path}")
                return WorkflowState(**data)
            except Exception:
                # Backup corrupted state
                timestamp = int(time.time())
                backup_path = self.state_path.with_suffix(f".bak.{timestamp}.json")
                try:
                    shutil.move(str(self.state_path), str(backup_path))
                    logger.exception(
                        "Failed to load state. Corrupted file moved to %s. Starting fresh.",
                        backup_path
                    )
                except Exception:
                    logger.exception("Failed to load state and failed to backup. Starting fresh.")

        logger.info("Initializing new workflow state.")
        return WorkflowState(
            cycle_index=0,
            current_phase=WorkflowPhase.INITIALIZATION
        )

    def save(self, state: WorkflowState) -> None:
        """Persist current state to disk."""
        try:
            io.save_json(state.model_dump(mode="json"), self.state_path)
            logger.debug(f"Saved workflow state to {self.state_path}")
        except Exception:
            logger.exception("Failed to save state.")
