from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.calculation import DFTConfig


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


class OrchestratorConfig(BaseModel):
    """Placeholder for Orchestrator configuration."""
    model_config = ConfigDict(extra="forbid")
    # No fields defined in Cycle 01 spec yet, allowing empty or defaults if needed later.


class LammpsConfig(BaseModel):
    """Configuration for the LAMMPS executable and runtime environment."""
    model_config = ConfigDict(extra="forbid")

    command: str = "lmp_serial"
    timeout: int = 3600
    use_mpi: bool = False
    mpi_command: str = "mpirun -np 4"


class MDConfig(BaseModel):
    """Configuration for MD simulation parameters."""
    model_config = ConfigDict(extra="forbid")

    temperature: float
    pressure: Optional[float] = None
    n_steps: int
    timestep: float = 0.001
    ensemble: Literal["NVT", "NPT"]


# Alias for backward compatibility if needed, though we should update usages
MDParams = MDConfig


class StructureGenConfig(BaseModel):
    """Configuration for structure generation."""
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["bulk"] = "bulk"
    element: str
    crystal_structure: str
    lattice_constant: float
    rattle_stdev: float = 0.0
    supercell: tuple[int, int, int] = (1, 1, 1)


class TrainingConfig(BaseModel):
    """Placeholder for Training Configuration (Cycle 04)."""
    model_config = ConfigDict(extra="forbid")
    initial_potential: Optional[Path] = None
    # Additional fields to be added in Cycle 04


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)

    # New configurations
    structure_gen: StructureGenConfig
    md: MDConfig

    # Optional placeholders for future cycles
    dft: Optional[DFTConfig] = None
    training: Optional[TrainingConfig] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io
        data = io.load_yaml(path)
        return cls(**data)
