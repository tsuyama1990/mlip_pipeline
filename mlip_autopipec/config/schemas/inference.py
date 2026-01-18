from pathlib import Path
from typing import Literal, List
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator
import os

class InferenceConfig(BaseModel):
    temperature: float = Field(..., gt=0)
    pressure: float = Field(0.0, ge=0)  # 0 for NVT
    timestep: float = Field(0.001, gt=0)  # ps
    steps: int = Field(10000, gt=0)
    ensemble: Literal["nvt", "npt"] = "nvt"
    uq_threshold: float = Field(5.0, gt=0)
    sampling_interval: int = Field(100, gt=0)
    potential_path: FilePath
    lammps_executable: FilePath | None = None
    elements: List[str] = Field(default_factory=lambda: ["Al"]) # Default to Al for compatibility

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

class InferenceResult(BaseModel):
    succeeded: bool
    final_structure: Path
    uncertain_structures: List[Path]  # Paths to extracted frames
    max_gamma_observed: float

    model_config = ConfigDict(extra="forbid")
