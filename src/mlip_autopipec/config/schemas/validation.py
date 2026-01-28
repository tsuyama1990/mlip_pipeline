from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PhononConfig(BaseModel):
    """Configuration for Phonon validation."""

    enabled: bool = Field(default=True, description="Enable Phonon validation")
    supercell_matrix: list[int] = Field(
        default=[2, 2, 2],
        min_length=3,
        max_length=3,
        description="Supercell dimensions [Nx, Ny, Nz]",
    )
    displacement: float = Field(default=0.01, gt=0.0, description="Atomic displacement distance")
    symprec: float = Field(default=1e-5, gt=0.0, description="Symmetry precision")
    model_config = ConfigDict(extra="forbid")


class ElasticConfig(BaseModel):
    """Configuration for Elasticity validation."""

    enabled: bool = Field(default=True, description="Enable Elasticity validation")
    num_points: int = Field(default=5, ge=3, description="Number of distortion points per branch")
    max_distortion: float = Field(default=0.05, gt=0.0, description="Maximum distortion percentage")
    model_config = ConfigDict(extra="forbid")


class EOSConfig(BaseModel):
    """Configuration for Equation of State validation."""

    enabled: bool = Field(default=True, description="Enable EOS validation")
    num_points: int = Field(default=10, ge=5, description="Number of volume scaling points")
    strain_max: float = Field(default=0.1, gt=0.0, description="Maximum volumetric strain (+/-)")
    model_config = ConfigDict(extra="forbid")


class ValidationConfig(BaseModel):
    """
    Configuration for Physics Validation Suite.
    """

    phonon: PhononConfig = Field(default_factory=lambda: PhononConfig())
    elastic: ElasticConfig = Field(default_factory=lambda: ElasticConfig())
    eos: EOSConfig = Field(default_factory=lambda: EOSConfig())

    reference_data: dict[str, Any] | None = Field(
        default=None, description="Optional reference data (DFT/Exp) for comparison"
    )

    # Global settings
    fail_on_instability: bool = Field(
        default=False, description="Whether to raise error if instability detected"
    )

    model_config = ConfigDict(extra="forbid")
