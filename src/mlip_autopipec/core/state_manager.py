import json
import shutil
from pathlib import Path

from mlip_autopipec.domain_models import WorkflowState


class StateManager:
    """
    Manages the persistence of the workflow state.
    Ensures atomic writes to prevent data corruption.
    """

    def __init__(self, state_file: Path) -> None:
        """
        Args:
            state_file: Path to the JSON file where state is stored.
        """
        self.state_file = state_file

    def load(self) -> WorkflowState:
        """
        Loads the workflow state from disk.
        If the file does not exist, returns a fresh WorkflowState.

        Returns:
            The loaded or new WorkflowState.
        """
        if not self.state_file.exists():
            return WorkflowState()
        with self.state_file.open('r') as f:
            data = json.load(f)
        return WorkflowState.model_validate(data)

    def save(self, state: WorkflowState) -> None:
        """
        Saves the workflow state to disk atomically.

        Args:
            state: The WorkflowState object to save.
        """
        temp_file = self.state_file.with_suffix('.tmp')
        with temp_file.open('w') as f:
            json.dump(state.model_dump(), f, indent=2)
        shutil.move(str(temp_file), str(self.state_file))

    def cleanup(self) -> None:
        """
        Removes any temporary state files left over from failed saves.
        """
        temp_file = self.state_file.with_suffix('.tmp')
        if temp_file.exists():
            temp_file.unlink()
