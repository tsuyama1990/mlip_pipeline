from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Potential(BaseModel):
    """
    Represents a machine learning potential artifact (e.g., .yace file).
    Includes metadata about its lineage and performance.
    """
    model_config = ConfigDict(extra="forbid")

    path: Path = Field(..., description="Path to the potential file.")
    iteration: int = Field(..., ge=0, description="The active learning cycle iteration where this potential was generated.")
    parent_potential_path: Path | None = Field(None, description="Path to the potential used as a starting point (if any).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g. training metrics, generation id).")
