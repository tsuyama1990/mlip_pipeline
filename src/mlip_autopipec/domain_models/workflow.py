from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowState(BaseModel):
    iteration: int = 0
    current_potential_path: Path | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
