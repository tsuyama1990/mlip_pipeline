from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class WorkflowState(BaseModel):
    current_cycle: int = 0
    status: Literal["IDLE", "RUNNING", "STOPPED", "ERROR"] = "IDLE"


class StateManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = self._load()

    def _load(self) -> WorkflowState:
        if self.path.exists():
            with self.path.open("r") as f:
                content = f.read()
                if content.strip():
                    return WorkflowState.model_validate_json(content)
        return WorkflowState()

    def save(self) -> None:
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def update_cycle(self, cycle: int) -> None:
        self.state.current_cycle = cycle
        self.save()

    def update_status(self, status: Literal["IDLE", "RUNNING", "STOPPED", "ERROR"]) -> None:
        self.state.status = status
        self.save()
