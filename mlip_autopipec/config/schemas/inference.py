from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InferenceConfig(BaseModel):
    """Configuration for LAMMPS inference."""
    lammps_executable: Path | None = Field(None, description="Path to LAMMPS executable")
    temperature: float = Field(300.0, ge=0.0, description="MD temperature in Kelvin (K)")
    pressure: float = Field(1.0, ge=0.0, description="MD pressure in Bar")
    timestep: float = Field(1.0, gt=0.0, description="Timestep in fs")
    steps: int = Field(1000, gt=0, description="Number of MD steps")
    uncertainty_threshold: float = Field(5.0, gt=0.0, description="Max extrapolation grade before stop")

    model_config = ConfigDict(extra="forbid")

    @field_validator("lammps_executable")
    @classmethod
    def validate_executable(cls, v: Path | None) -> Path | None:
        """
        Validates that the executable path string is not empty if provided.
        Actual existence check is runtime dependent (e.g. remote nodes),
        but we can check if it's a valid path structure.
        """
        if v is not None:
            if str(v).strip() == "":
                raise ValueError("LAMMPS executable path cannot be empty.")
            # Optional: Check if it's absolute or in path?
            # shutil.which might fail if run in environment different from execution env.
            # So we stick to basic validation.
        return v


class InferenceResult(BaseModel):
    succeeded: bool
    final_structure: Path | None = None
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
