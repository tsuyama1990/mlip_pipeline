from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .structure import Structure


class Potential(BaseModel):
    """
    Represents a trained potential model.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    version: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ExplorationResult(BaseModel):
    """
    Result of an exploration run (Molecular Dynamics).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["converged", "halted"]
    structure: Structure | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """
    Result of a validation run.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    report_path: Path | None = None
    details: dict[str, Any] = Field(default_factory=dict)
