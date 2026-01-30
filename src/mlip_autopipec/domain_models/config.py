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

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v


class LammpsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = "lmp_serial"
    timeout: int = 3600
    use_mpi: bool = False
    mpi_command: str = "mpirun"


class StructureGenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element: str = "Si"
    crystal_structure: str = "diamond"
    lattice_constant: float = 5.43
    supercell: tuple[int, int, int] = (2, 2, 2)
    rattle_stdev: float = 0.0


class OrchestratorConfig(BaseModel):
    """Placeholder for Orchestrator configuration."""

    model_config = ConfigDict(extra="forbid")
    # No fields defined in Cycle 01 spec yet, allowing empty or defaults if needed later.


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)
    structure_gen: StructureGenConfig = Field(default_factory=StructureGenConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io

        data = io.load_yaml(path)
        return cls(**data)
