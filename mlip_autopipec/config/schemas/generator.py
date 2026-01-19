from pydantic import BaseModel, ConfigDict, Field, field_validator


class SQSConfig(BaseModel):
    enabled: bool = Field(True, description="Enable SQS generation for bulk alloys.")
    supercell_matrix: list[list[int]] = Field(
        default=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        description="Transformation matrix for the supercell (3x3).",
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("supercell_matrix")
    def validate_matrix_shape(cls, v: list[list[int]]) -> list[list[int]]:  # noqa: N805
        if len(v) != 3:
            msg = "Supercell matrix must have 3 rows."
            raise ValueError(msg)
        for row in v:
            if len(row) != 3:
                msg = "Supercell matrix rows must have 3 elements."
                raise ValueError(msg)
        return v


class DistortionConfig(BaseModel):
    enabled: bool = Field(True, description="Enable strain and rattle distortions.")
    rattling_amplitude: float = Field(
        0.05, gt=0, description="Standard deviation for Gaussian rattling (Angstroms)."
    )
    strain_range: tuple[float, float] = Field(
        (-0.05, 0.05), description="Min and max linear strain."
    )
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
    interstitial_elements: list[str] = Field(
        default_factory=list, description="Elements to insert as interstitials."
    )
    model_config = ConfigDict(extra="forbid")


class GeneratorConfig(BaseModel):
    # Required fields to ensure explicit configuration
    sqs: SQSConfig = Field(..., description="SQS Configuration")
    distortion: DistortionConfig = Field(..., description="Distortion Configuration")
    nms: NMSConfig = Field(..., description="Normal Mode Sampling Configuration")
    defects: DefectConfig = Field(..., description="Defect Configuration")

    model_config = ConfigDict(extra="forbid")
