import json
from pathlib import Path

from mlip_autopipec.core.exceptions import StateError
from mlip_autopipec.domain_models.inputs import ProjectState


class StateManager:
    """Manages the persistent state of the workflow."""

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.state_file = self.work_dir / "workflow_state.json"

    def load(self) -> ProjectState:
        """Load the workflow state from disk or create a new one."""
        if not self.state_file.exists():
            return ProjectState()

        try:
            with self.state_file.open("r") as f:
                data = json.load(f)
            return ProjectState.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise StateError(f"Failed to load state from {self.state_file}: {e}") from e

    def save(self, state: ProjectState) -> None:
        """Save the workflow state atomically."""
        try:
            # Atomic write pattern: write to temp file, then rename
            temp_file = self.state_file.with_suffix(".tmp")

            # Ensure work_dir exists
            self.work_dir.mkdir(parents=True, exist_ok=True)

            with temp_file.open("w") as f:
                f.write(state.model_dump_json(indent=2))

            # Atomic rename (replace)
            temp_file.replace(self.state_file)

        except OSError as e:
            raise StateError(f"Failed to save state to {self.state_file}: {e}") from e
