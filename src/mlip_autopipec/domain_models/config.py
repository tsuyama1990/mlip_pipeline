from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_path: Path = Path("mlip_pipeline.log")


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str]
    cutoff: float
    seed: int = 42

    # Structure Generation
    lattice_constant: float = 5.43
    crystal_structure: str = "diamond"

    # LAMMPS Potential
    pair_style: str = "lj/cut 2.5"
    pair_coeff: list[str] = ["* * 1.0 1.0 2.5"]

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
    """Parameters for MD simulation."""
    model_config = ConfigDict(extra="forbid")

    temperature: float = 300.0
    pressure: float = 0.0
    n_steps: int = 1000
    timestep: float = 0.001  # ps


class LammpsConfig(BaseModel):
    """Configuration for LAMMPS execution."""
    model_config = ConfigDict(extra="forbid")

    command: str = "lmp_serial"
    cores: int = 1
    timeout: float = 3600.0
    base_work_dir: Path = Path("_work_md")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)
    md_params: MDParams = Field(default_factory=MDParams)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io
        data = io.load_yaml(path)
        return cls(**data)
