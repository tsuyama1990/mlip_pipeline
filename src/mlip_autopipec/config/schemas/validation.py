from pydantic import BaseModel, ConfigDict, Field


class PhononConfig(BaseModel):
    """Configuration for Phonon validation."""

    supercell_matrix: list[int] = Field(
        default=[2, 2, 2],
        min_length=3,
        max_length=3,
        description="Supercell dimensions [Nx, Ny, Nz]",
    )
    displacement: float = Field(0.01, gt=0.0, description="Atomic displacement distance")
    symprec: float = Field(1e-5, gt=0.0, description="Symmetry precision")
    model_config = ConfigDict(extra="forbid")


class ElasticConfig(BaseModel):
    """Configuration for Elasticity validation."""

    num_points: int = Field(5, ge=3, description="Number of distortion points per branch")
    max_distortion: float = Field(0.05, gt=0.0, description="Maximum distortion percentage")
    model_config = ConfigDict(extra="forbid")


class EOSConfig(BaseModel):
    """Configuration for Equation of State validation."""

    num_points: int = Field(10, ge=5, description="Number of volume scaling points")
    strain_max: float = Field(0.1, gt=0.0, description="Maximum volumetric strain (+/-)")
    model_config = ConfigDict(extra="forbid")


class ValidationConfig(BaseModel):
    """
    Configuration for Physics Validation Suite.
    """

    phonon: PhononConfig = Field(default_factory=PhononConfig)
    elastic: ElasticConfig = Field(default_factory=ElasticConfig)
    eos: EOSConfig = Field(default_factory=EOSConfig)

    # Global settings
    fail_on_instability: bool = Field(False, description="Whether to raise error if instability detected")

    model_config = ConfigDict(extra="forbid")
