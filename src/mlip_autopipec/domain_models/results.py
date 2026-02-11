from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CalculationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy: float | None = Field(None, description="Total potential energy (eV)")
    forces: list[list[float]] | None = Field(None, description="Nx3 array of atomic forces (eV/A)")
    stress: list[list[float]] | None = Field(None, description="3x3 stress tensor (eV/A^3)")
    virial: list[list[float]] | None = Field(None, description="3x3 virial tensor")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional calculation metadata")


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    potential_path: Path = Field(..., description="Path to the trained potential file")
    metrics: dict[str, float] = Field(default_factory=dict, description="Validation metrics (RMSE, MAE)")
    history: list[dict[str, float]] = Field(default_factory=list, description="Training history/loss curve")
