from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HistoryEntry(BaseModel):  # type: ignore[misc]
    iteration: int
    potential_path: str
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    candidates_count: int = 0
    new_data_count: int = 0

    model_config = ConfigDict(extra="forbid")


class WorkflowState(BaseModel):  # type: ignore[misc]
    iteration: int = 0
    current_potential_path: Path | None = None
    history: list[HistoryEntry] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
