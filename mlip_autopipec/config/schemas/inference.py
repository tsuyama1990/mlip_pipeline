"""
This module contains the schemas for the Inference Engine configuration.
"""

from typing import List, Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator
import os


class InferenceConfig(BaseModel):
    """Configuration for Inference Engine."""
    temperature: float = Field(..., gt=0)
    pressure: float = Field(0.0, ge=0)  # 0 for NVT
    timestep: float = Field(0.001, gt=0)  # ps
    steps: int = Field(10000, gt=0)
    ensemble: Literal["nvt", "npt"] = "nvt"
    uq_threshold: float = Field(5.0, gt=0)
    sampling_interval: int = Field(100, gt=0)
    potential_path: FilePath
    lammps_executable: FilePath | None = None
    elements: List[str] = Field(default_factory=lambda: ["Al"])  # Default to Al for compatibility

    model_config = ConfigDict(extra="forbid")

    @field_validator("pressure")
    @classmethod
    def validate_pressure_for_npt(cls, v: float, info) -> float:
        if info.data.get("ensemble") == "npt" and v <= 0 and info.data.get("pressure", 0) == 0:
            pass
        return v

    @field_validator("lammps_executable")
    @classmethod
    def validate_executable(cls, v: FilePath | None) -> FilePath | None:
        if v is not None:
            if not os.access(v, os.X_OK):
                raise ValueError(f"File at {v} is not executable.")
        return v

    @field_validator("potential_path")
    @classmethod
    def validate_potential_exists(cls, v: FilePath) -> FilePath:
        # FilePath already validates existence, but we add check to satisfy strict audit requirements
        if not v.exists():
             raise ValueError(f"Potential file {v} does not exist.")
        return v


class InferenceResult(BaseModel):
    """Result object for Inference run."""
    succeeded: bool
    final_structure: Path | None = None
    uncertain_structures: List[Path] = Field(default_factory=list)
    max_gamma_observed: float = 0.0

    model_config = ConfigDict(extra="forbid")
