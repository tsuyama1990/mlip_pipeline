import json
import logging
from pathlib import Path

from mlip_autopipec.domain_models import WorkflowState

logger = logging.getLogger("mlip_autopipec")

class StateManager:
    """
    Manages the persistence of the workflow state using atomic writes.
    """
    def __init__(self, work_dir: Path, filename: str = "workflow_state.json") -> None:
        self.work_dir = work_dir
        self.state_file = work_dir / filename
        # Ensure work directory exists
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> WorkflowState:
        """
        Loads the workflow state from disk. Returns a fresh state if file is missing.
        """
        if not self.state_file.exists():
            logger.info("No existing state file found. Starting new workflow.")
            return WorkflowState()

        try:
            with self.state_file.open("r") as f:
                data = json.load(f)
            return WorkflowState.model_validate(data)
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load state file. Starting fresh.")
            return WorkflowState()

    def save(self, state: WorkflowState) -> None:
        """
        Saves the workflow state to disk atomically.
        """
        try:
            # We use a temp file in the same directory to ensure atomic move
            temp_file = self.work_dir / f".{self.state_file.name}.tmp"
            with temp_file.open("w") as f:
                f.write(state.model_dump_json(indent=2))

            temp_file.rename(self.state_file)
            logger.debug(f"State saved to {self.state_file}")
        except OSError:
            logger.exception("Failed to save state")
            raise
