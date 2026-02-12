import json
from pathlib import Path

from mlip_autopipec.domain_models.datastructures import WorkflowState


class StateManager:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.state_file = work_dir / "workflow_state.json"

    def load_state(self) -> WorkflowState | None:
        if not self.state_file.exists():
            return None
        with self.state_file.open("r") as f:
            data = json.load(f)
        return WorkflowState.model_validate(data)

    def save_state(self, state: WorkflowState) -> None:
        tmp_file = self.state_file.with_suffix(".tmp")
        with tmp_file.open("w") as f:
            f.write(state.model_dump_json(indent=2))
        tmp_file.rename(self.state_file)

    def cleanup(self) -> None:
        for f in self.work_dir.glob("*.tmp"):
            f.unlink()
