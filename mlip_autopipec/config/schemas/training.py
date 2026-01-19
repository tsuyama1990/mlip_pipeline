import math
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
        if v is not None and not os.access(v, os.X_OK):
            msg = f"File at {v} is not executable."
            raise ValueError(msg)
        return v


class TrainConfig(BaseModel):
    cutoff: float = Field(6.0, gt=0)  # Angstroms
    loss_weights: dict[str, float] = Field(default={"energy": 1.0, "forces": 100.0, "stress": 10.0})
    test_fraction: float = Field(0.1, ge=0.0, lt=1.0)
    max_generations: int = Field(10, ge=1)
    enable_delta_learning: bool = True
    batch_size: int = Field(100, gt=0)
    ace_basis_size: str = Field("medium", pattern="^(small|medium|large)$")
    model_config = ConfigDict(extra="forbid")


class TrainingResult(BaseModel):
    potential_path: Path
    rmse_energy: float = Field(..., ge=0)
    rmse_forces: float = Field(..., ge=0)
    training_time: float = Field(..., ge=0)
    generation: int = Field(..., ge=0)
    model_config = ConfigDict(extra="forbid")


class TrainingData(BaseModel):
    energy: float
    forces: list[list[float]]
    model_config = ConfigDict(extra="forbid")

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            return v
        if not all(len(row) == 3 for row in v):
            msg = "Each force vector must have 3 components (x, y, z)."
            raise ValueError(msg)
        # Check for NaN or Infinity
        for row in v:
            for val in row:
                if not math.isfinite(val):
                    msg = f"Force value {val} is not finite."
                    raise ValueError(msg)
        return v


class TrainingRunMetrics(BaseModel):
    generation: int
    num_structures: int
    rmse_forces: float
    rmse_energy_per_atom: float
    model_config = ConfigDict(extra="forbid")
