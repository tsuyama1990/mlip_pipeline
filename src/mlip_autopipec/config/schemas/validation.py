from pydantic import BaseModel, ConfigDict, Field


class PhononConfig(BaseModel):
    enabled: bool = True
    supercell_matrix: list[list[int]] = Field(
        default_factory=lambda: [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    displacement: float = 0.01
    mesh: list[int] = Field(default_factory=lambda: [20, 20, 20])
    plot: bool = False

    model_config = ConfigDict(extra="forbid")


class ElasticityConfig(BaseModel):
    enabled: bool = True
    strain_max: float = 0.05
    num_points: int = 5

    model_config = ConfigDict(extra="forbid")


class EOSConfig(BaseModel):
    enabled: bool = True
    strain_max: float = 0.10
    num_points: int = 7

    model_config = ConfigDict(extra="forbid")


class ValidationConfig(BaseModel):
    phonon: PhononConfig = Field(default_factory=PhononConfig)
    elasticity: ElasticityConfig = Field(default_factory=ElasticityConfig)
    eos: EOSConfig = Field(default_factory=EOSConfig)

    model_config = ConfigDict(extra="forbid")
