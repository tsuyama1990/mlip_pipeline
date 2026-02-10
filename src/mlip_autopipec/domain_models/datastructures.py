from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from .enums import WorkflowStage


def sanitize_value(v: Any) -> Any:
    """Recursively convert numpy types to native python types."""
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {k: sanitize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [sanitize_value(val) for val in v]
    return v

class Structure(BaseModel):
    """
    A wrapper around ASE Atoms with additional metadata.
    """
    atoms: Atoms
    provenance: str = "unknown"
    tags: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __init__(self, **data: Any) -> None:
        if "tags" in data:
            data["tags"] = sanitize_value(data["tags"])
        super().__init__(**data)

class Dataset(BaseModel):
    """
    Represents a dataset stored on disk.
    """
    path: Path
    format: str = "extxyz"

    model_config = ConfigDict(extra="forbid")

class Potential(BaseModel):
    """
    Represents a fitted potential.
    """
    path: Path
    type: str

    model_config = ConfigDict(extra="forbid")

class WorkflowState(BaseModel):
    """
    Tracks the current state of the active learning loop.
    """
    iteration: int = 0
    current_stage: WorkflowStage = WorkflowStage.EXPLORE
    latest_potential_path: Path | None = None

    model_config = ConfigDict(extra="forbid")
