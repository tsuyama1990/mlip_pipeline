from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_path: Path = Path("mlip_pipeline.log")


class ElementParams(BaseModel):
    """Physical parameters for a specific chemical element."""
    model_config = ConfigDict(extra="forbid")

    mass: float
    lj_sigma: float
    lj_epsilon: float
    zbl_z: int


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str]
    cutoff: float
    seed: int = 42
    element_params: dict[str, ElementParams] = Field(default_factory=dict)

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v


class OrchestratorConfig(BaseModel):
    """Placeholder for Orchestrator configuration."""
    model_config = ConfigDict(extra="forbid")
    # No fields defined in Cycle 01 spec yet, allowing empty or defaults if needed later.


class MDParams(BaseModel):
    """Parameters for Molecular Dynamics simulation."""
    model_config = ConfigDict(extra="forbid")

    temperature: float = 300.0
    n_steps: int = 1000
    timestep: float = 0.001  # ps
    ensemble: str = "nvt"  # nvt, npt
    pressure: Optional[float] = None  # bar, for NPT


class ExplorationConfig(BaseModel):
    """Configuration for the exploration phase."""
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["random", "template"] = "random"
    supercell_size: list[int] = Field(default_factory=lambda: [1, 1, 1])
    rattle_amplitude: float = 0.0
    num_candidates: int = 1
    composition: Optional[str] = None
    lattice_constant: float = 5.43
    md_params: MDParams = Field(default_factory=MDParams)


class LammpsConfig(BaseModel):
    """Configuration for LAMMPS execution."""
    model_config = ConfigDict(extra="forbid")

    command: str = "lmp_serial"
    timeout: int = 3600
    cores: int = 1


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig
    structure_gen: ExplorationConfig = Field(default_factory=ExplorationConfig)
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io
        data = io.load_yaml(path)
        return cls(**data)
