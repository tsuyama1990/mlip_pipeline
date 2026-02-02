from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowState(BaseModel):
    """
    Tracks the state of the Active Learning Cycle.
    Designed to be serialized to JSON for idempotency.
    """
    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(0, ge=0, description="Current cycle iteration number.")
    current_potential_path: Path | None = Field(None, description="Path to the latest accepted potential.")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Log of metrics/events from previous cycles.")

    def update_potential(self, potential_path: Path) -> None:
        """Updates the current potential path."""
        self.current_potential_path = potential_path

    def increment_iteration(self) -> None:
        """Increments the iteration counter."""
        self.iteration += 1
