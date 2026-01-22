from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class InferenceConfig(BaseModel):
    """Configuration for LAMMPS inference."""
    lammps_executable: Path | None = Field(None, description="Path to LAMMPS executable")
    temperature: float = Field(300.0, ge=0.0, description="MD temperature in Kelvin")
    pressure: float = Field(1.0, ge=0.0, description="MD pressure in Bar")
    timestep: float = Field(1.0, gt=0.0, description="Timestep in fs")
    steps: int = Field(1000, gt=0, description="Number of MD steps")
    uncertainty_threshold: float = Field(5.0, gt=0.0, description="Max extrapolation grade before stop")

    model_config = ConfigDict(extra="forbid")


class InferenceResult(BaseModel):
    succeeded: bool
    final_structure: Path | None
    uncertain_structures: list[Path]
    max_gamma_observed: float
    model_config = ConfigDict(extra="forbid")


class EmbeddingConfig(BaseModel):
    """Configuration for cluster embedding."""
    core_radius: float = Field(4.0, gt=0.0)
    buffer_width: float = Field(2.0, gt=0.0)
    model_config = ConfigDict(extra="forbid")

    @property
    def box_size(self) -> float:
        return 2 * (self.core_radius + self.buffer_width) + 2.0
