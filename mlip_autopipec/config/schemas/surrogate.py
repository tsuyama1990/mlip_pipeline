from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DescriptorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r_cut: float = Field(default=6.0, gt=0.0, description="Cutoff radius in Angstroms")
    n_max: int = Field(default=8, ge=1, description="Number of radial basis functions")
    l_max: int = Field(default=6, ge=0, description="Maximum degree of spherical harmonics")
    sigma: float = Field(default=0.5, gt=0.0, description="Standard deviation of the gaussians")
    descriptor_type: Literal["soap", "ace"] = Field(
        default="soap", description="Type of descriptor"
    )


class SurrogateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(default="medium", description="MACE model size or path")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device to run inference on")
    fps_n_samples: int = Field(default=100, ge=1, description="Number of samples to select via FPS")
    force_threshold: float = Field(
        default=50.0, gt=0.0, description="Force threshold in eV/A for filtering"
    )
    descriptor_config: DescriptorConfig = Field(
        default_factory=DescriptorConfig, description="Descriptor configuration"
    )


class SelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_indices: list[int] = Field(description="Indices of selected structures")
    scores: list[float] = Field(description="Distance scores for selected structures")


class RejectionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., description="Index of the rejected structure")
    max_force: float = Field(..., description="Maximum force encountered in the structure")
    reason: str = Field(..., description="Reason for rejection")


class DescriptorResult(BaseModel):
    """
    Wraps the output of descriptor calculation to ensure data integrity.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    features: Any = Field(..., description="Numpy array of shape (N_structures, N_features)")

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Any) -> Any:
        import numpy as np

        if not isinstance(v, np.ndarray):
            raise TypeError("Features must be a numpy array.")
        if v.ndim != 2:
            raise ValueError(f"Features must be 2D array, got {v.ndim}D.")
        return v
