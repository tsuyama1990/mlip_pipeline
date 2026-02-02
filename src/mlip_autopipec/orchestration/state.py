import json
from pathlib import Path

from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.utils.file_ops import atomic_write


class StateManager:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def save(self, state: WorkflowState) -> None:
        """Saves the workflow state atomically."""
        with atomic_write(self.state_file) as temp_path, temp_path.open("w") as f:
            f.write(state.model_dump_json(indent=2))

    def load(self) -> WorkflowState | None:
        """Loads the workflow state from file."""
        if not self.state_file.exists():
            return None

        with self.state_file.open("r") as f:
            data = json.load(f)
        return WorkflowState.model_validate(data)
