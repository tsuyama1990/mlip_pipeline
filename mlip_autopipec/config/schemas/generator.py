
from pydantic import BaseModel, ConfigDict, Field


class GeneratorConfig(BaseModel):
    # SQS settings
    supercell_matrix: list[list[int]] = Field(default=[[2,0,0], [0,2,0], [0,0,2]])

    # Distortion settings
    rattling_amplitude: float = Field(0.05, gt=0)  # Angstroms
    strain_range: tuple[float, float] = Field((-0.05, 0.05)) # +/- 5%
    n_strain_steps: int = Field(5, ge=1)
    n_rattle_steps: int = Field(3, ge=1)

    # NMS settings
    temperatures: list[int] = Field(default=[300, 600, 1000]) # Kelvin

    model_config = ConfigDict(extra="forbid")
