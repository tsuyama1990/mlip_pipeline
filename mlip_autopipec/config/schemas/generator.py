from pydantic import BaseModel, ConfigDict, Field


class SQSConfig(BaseModel):
    enabled: bool = Field(True, description="Enable SQS generation for bulk alloys.")
    supercell_matrix: list[list[int]] = Field(
        default=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        description="Transformation matrix for the supercell."
    )
    model_config = ConfigDict(extra="forbid")


class DistortionConfig(BaseModel):
    enabled: bool = Field(True, description="Enable strain and rattle distortions.")
    rattling_amplitude: float = Field(0.05, gt=0, description="Standard deviation for Gaussian rattling (Angstroms).")
    strain_range: tuple[float, float] = Field((-0.05, 0.05), description="Min and max linear strain.")
    n_strain_steps: int = Field(5, ge=1, description="Number of strain steps.")
    n_rattle_steps: int = Field(3, ge=1, description="Number of rattle steps per structure.")
    model_config = ConfigDict(extra="forbid")


class NMSConfig(BaseModel):
    enabled: bool = Field(True, description="Enable Normal Mode Sampling for molecules.")
    temperatures: list[int] = Field(default=[300, 600, 1000], description="Temperatures for NMS.")
    n_samples: int = Field(5, ge=1, description="Number of samples per temperature.")
    model_config = ConfigDict(extra="forbid")


class DefectConfig(BaseModel):
    enabled: bool = Field(False, description="Enable defect generation.")
    vacancies: bool = Field(True, description="Generate vacancies.")
    interstitials: bool = Field(False, description="Generate interstitials.")
    interstitial_elements: list[str] = Field(default_factory=list, description="Elements to insert as interstitials.")
    model_config = ConfigDict(extra="forbid")


class GeneratorConfig(BaseModel):
    sqs: SQSConfig = Field(default_factory=SQSConfig)
    distortion: DistortionConfig = Field(default_factory=DistortionConfig)
    nms: NMSConfig = Field(default_factory=NMSConfig)
    defects: DefectConfig = Field(default_factory=DefectConfig)

    # Backwards compatibility / shortcuts (Optional)
    # If we want to keep the flat structure for simplicity in some tests, we might need a validator.
    # But for strict architecture, let's enforce hierarchy.

    model_config = ConfigDict(extra="forbid")
