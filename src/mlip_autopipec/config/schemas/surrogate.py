from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SurrogateConfig(BaseModel):
    """Configuration for surrogate model selection."""

    model_type: str = Field(
        "mace_mp", description="Type of surrogate model: 'mace_mp', 'chgnet', or 'mock'"
    )
    model_path: str = Field(
        "medium", description="Path to weights or HuggingFace ID (e.g., 'medium' for MACE)"
    )
    device: str = Field("cpu", description="Device to run on: 'cuda' or 'cpu'")
    force_threshold: float = Field(50.0, description="Max allowed force (eV/A) before rejection")
    n_samples: int = Field(100, description="Number of structures to select via FPS")

    model_config = ConfigDict(extra="forbid")


class RejectionInfo(BaseModel):
    """Information about a rejected structure."""

    index: int
    max_force: float
    reason: str

    model_config = ConfigDict(extra="forbid")


class DescriptorConfig(BaseModel):
    """Configuration for descriptor calculation (e.g. SOAP)."""

    r_cut: float = Field(5.0, description="Cutoff radius")
    n_max: int = Field(4, description="Number of radial basis functions")
    l_max: int = Field(4, description="Max degree of spherical harmonics")
    sigma: float = Field(0.5, description="Gaussian width")

    model_config = ConfigDict(extra="forbid")


class DescriptorResult(BaseModel):
    """Result of descriptor calculation."""

    features: Any = Field(..., description="Numpy array of features")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
