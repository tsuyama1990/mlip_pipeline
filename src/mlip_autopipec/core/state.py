import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="ignore")
    current_cycle: int = 0
    status: str = "IDLE"
    last_updated: str = ""

    def __repr__(self) -> str:
        return f"<WorkflowState(cycle={self.current_cycle}, status={self.status})>"

    def __str__(self) -> str:
        return f"WorkflowState(cycle={self.current_cycle}, status={self.status})"


class StateManager:
    """Manages the persistent state of the pipeline."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> WorkflowState:
        if not self.state_file.exists():
            return WorkflowState()
        try:
            with self.state_file.open("r") as f:
                data = json.load(f)
            return WorkflowState.model_validate(data)
        except Exception:
            return WorkflowState()

    def _save_state(self) -> None:
        with self.state_file.open("w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def update_cycle(self, cycle: int) -> None:
        self.state.current_cycle = cycle
        self._save_state()

    def update_status(self, status: str) -> None:
        self.state.status = status
        self._save_state()

    def __repr__(self) -> str:
        return f"<StateManager(file={self.state_file}, state={self.state})>"

    def __str__(self) -> str:
        return f"StateManager({self.state_file.name})"
