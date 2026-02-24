"""State management domain models."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineState(BaseModel):
    """Represents the state of the active learning pipeline."""

    model_config = ConfigDict(extra="forbid")

    current_step: int = Field(
        ...,
        description="The current step number being executed (1-7).",
        ge=1,
        le=7,
    )
    completed_steps: list[int] = Field(
        default_factory=list,
        description="List of completed step numbers.",
    )
    artifacts: dict[str, Path] = Field(
        default_factory=dict,
        description="Dictionary of artifacts produced by steps (e.g. paths to datasets, potentials).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for state persistence (e.g. iteration counts).",
    )
