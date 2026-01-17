import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator


class LossWeights(BaseModel):
    energy: float = Field(1.0, gt=0)
    forces: float = Field(100.0, gt=0)
    stress: float = Field(10.0, gt=0)
    model_config = ConfigDict(extra="forbid")

class PacemakerLossWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy: float = Field(..., gt=0)
    forces: float = Field(..., gt=0)
    stress: float = Field(..., gt=0)

class PacemakerACEParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    radial_basis: str
    correlation_order: int = Field(..., ge=2)
    element_dependent_cutoffs: bool

class PacemakerFitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_filename: str
    loss_weights: PacemakerLossWeights
    ace: PacemakerACEParams

class PacemakerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fit_params: PacemakerFitParams

class TrainingConfig(BaseModel):
    pacemaker_executable: FilePath | None = None
    data_source_db: Path
    template_file: FilePath | None = None
    delta_learning: bool = True
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    ace_params: PacemakerACEParams = Field(
        default_factory=lambda: PacemakerACEParams(
            radial_basis="radial", correlation_order=3, element_dependent_cutoffs=False
        )
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("pacemaker_executable")
    @classmethod
    def validate_executable(cls, v: FilePath | None) -> FilePath | None:
        if v is not None:
            if not os.access(v, os.X_OK):
                raise ValueError(f"File at {v} is not executable.")
        return v

class TrainingData(BaseModel):
    energy: float
    forces: list[list[float]]
    model_config = ConfigDict(extra="forbid")

class TrainingRunMetrics(BaseModel):
    generation: int
    num_structures: int
    rmse_forces: float
    rmse_energy_per_atom: float
    model_config = ConfigDict(extra="forbid")
