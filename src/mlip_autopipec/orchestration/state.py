import json
from pathlib import Path

from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.utils.file_ops import atomic_write


class StateManager:
    """
    Manages persistence of the WorkflowState to disk using atomic writes.
    """
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def load(self) -> WorkflowState:
        """
        Loads the state from JSON. Returns a fresh state if file doesn't exist.
        """
        if not self.state_file.exists():
            return WorkflowState()

        with self.state_file.open("r") as f:
            data = json.load(f)
        return WorkflowState(**data)

    def save(self, state: WorkflowState) -> None:
        """
        Saves the state to JSON atomically.
        """
        # Pydantic v2 model_dump(mode='json') automatically handles Path serialization
        data = state.model_dump(mode='json')

        with atomic_write(self.state_file) as temp_path, temp_path.open("w") as f:
            json.dump(data, f, indent=2)
