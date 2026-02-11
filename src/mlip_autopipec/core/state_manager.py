import json
import shutil
from pathlib import Path

from mlip_autopipec.domain_models import WorkflowState


class StateManager:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def load(self) -> WorkflowState:
        if not self.state_file.exists():
            return WorkflowState()
        with self.state_file.open('r') as f:
            data = json.load(f)
        return WorkflowState.model_validate(data)

    def save(self, state: WorkflowState) -> None:
        temp_file = self.state_file.with_suffix('.tmp')
        with temp_file.open('w') as f:
            json.dump(state.model_dump(), f, indent=2)
        shutil.move(str(temp_file), str(self.state_file))
