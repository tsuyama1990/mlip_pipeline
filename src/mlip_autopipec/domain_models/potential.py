from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    path: Path
    generation_id: int
    parent_id: int | None = None
    type: str = "ace"  # e.g., "ace", "hybrid", "baseline"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
