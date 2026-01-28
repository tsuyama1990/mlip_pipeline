from pydantic import BaseModel, ConfigDict, Field


class SQSConfig(BaseModel):
    """Configuration for Special Quasirandom Structures (SQS)."""

    enabled: bool = Field(default=True, description="Whether to use SQS generation for alloys.")
    supercell_size: list[int] = Field(
        default=[2, 2, 2], description="Dimensions of the supercell (e.g., [2, 2, 2])."
    )
    model_config = ConfigDict(extra="forbid")


class DistortionConfig(BaseModel):
    """Configuration for lattice strain and atomic rattling."""

    enabled: bool = Field(default=True, description="Whether to apply distortions.")
    strain_range: tuple[float, float] = Field(
        default=(-0.05, 0.05), description="Range of linear strain (min, max)."
    )
    rattle_stdev: float = Field(
        default=0.01, description="Standard deviation of Gaussian noise for rattling (Angstroms)."
    )
    n_strain_steps: int = Field(default=5, description="Number of strain steps.")
    n_rattle_steps: int = Field(default=1, description="Number of rattle steps per strain.")
    rattling_amplitude: float = Field(
        default=0.01, description="Amplitude for rattling (deprecated, use rattle_stdev)."
    )

    model_config = ConfigDict(extra="forbid")


class DefectConfig(BaseModel):
    """Configuration for point defects."""

    enabled: bool = Field(default=False, description="Whether to generate defects.")
    vacancies: bool = Field(default=False, description="Generate vacancies.")
    interstitials: bool = Field(default=False, description="Generate interstitials.")
    interstitial_elements: list[str] = Field(
        default_factory=list, description="List of elements to insert as interstitials."
    )
    interstitial_min_dist: float = Field(
        default=1.4,
        gt=0.0,
        description="Minimum distance between interstitial and other atoms (Angstrom).",
    )
    interstitial_cluster_cutoff: float = Field(
        default=0.1,
        gt=0.0,
        description="Cutoff distance for clustering similar interstitial sites (Angstrom).",
    )
    model_config = ConfigDict(extra="forbid")


class GeneratorConfig(BaseModel):
    """
    Configuration for the physics-informed structure generator.
    Aggregates SQS, Distortion, and Defect configurations.
    """

    sqs: SQSConfig = Field(default_factory=SQSConfig)
    distortion: DistortionConfig = Field(default_factory=DistortionConfig)
    defects: DefectConfig = Field(default_factory=DefectConfig)
    number_of_structures: int = Field(
        default=10, description="Number of unique structures to generate per batch."
    )
    seed: int | None = Field(default=None, description="Random seed for deterministic generation.")

    model_config = ConfigDict(extra="forbid")
