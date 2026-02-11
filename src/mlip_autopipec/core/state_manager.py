import json
from pathlib import Path

from pydantic import ValidationError

from mlip_autopipec.domain_models import WorkflowState
from mlip_autopipec.utils.io import load_json, save_json


class StateManager:
    """
    Manages the persistence of the workflow state.
    """

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def load(self) -> WorkflowState:
        """
        Loads the workflow state from disk.
        If the file does not exist, returns a new WorkflowState (IDLE, iteration 0).
        """
        if not self.state_file.exists():
            return WorkflowState()

        try:
            data = load_json(self.state_file)
            return WorkflowState.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            msg = f"Failed to load state file {self.state_file}: {e}"
            raise RuntimeError(msg) from e

    def save(self, state: WorkflowState) -> None:
        """
        Saves the workflow state to disk using atomic write.
        """
        save_json(state.model_dump(mode="json"), self.state_file)

    def cleanup(self) -> None:
        """
        Removes temporary files generated during atomic writes.
        """
        temp_file = self.state_file.with_suffix(".tmp")
        if temp_file.exists():
            temp_file.unlink()
