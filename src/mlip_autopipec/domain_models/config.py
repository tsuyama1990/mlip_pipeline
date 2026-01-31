from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.calculation import DFTConfig
from mlip_autopipec.domain_models.dynamics import LammpsConfig, MDConfig
from mlip_autopipec.domain_models.training import TrainingConfig


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_path: Path = Path("mlip_pipeline.log")


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str]
    cutoff: float
    seed: int = 42

    # Hybrid Potential Settings (ACE + ZBL)
    pair_style: Literal["pace", "hybrid/overlay"] = "pace"
    zbl_inner_cutoff: float = 1.0
    zbl_outer_cutoff: float = 2.0

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator (Active Learning Loop)."""

    model_config = ConfigDict(extra="forbid")
    max_iterations: int = 1
    uncertainty_threshold: float = 5.0

    # Active Set Selection
    active_set_optimization: bool = True
    max_active_set_size: int = 1000


class BulkStructureGenConfig(BaseModel):
    """Configuration for bulk structure generation."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["bulk"] = "bulk"
    element: str
    crystal_structure: str
    lattice_constant: float
    rattle_stdev: float = 0.0
    supercell: tuple[int, int, int] = (1, 1, 1)


# Union of all structure generation configs
StructureGenConfig = Union[BulkStructureGenConfig]


class ValidationConfig(BaseModel):
    """Configuration for the Validation Framework."""

    model_config = ConfigDict(extra="forbid")

    # Phonon
    phonon_tolerance: float = -0.05  # THz, strictly negative to allow numeric noise
    phonon_supercell: tuple[int, int, int] = (2, 2, 2)

    # Elastic
    elastic_stability_tolerance: float = 1e-4  # GPa

    # EOS
    eos_vol_range: float = 0.1  # +/- 10%
    eos_n_points: int = 10

    # Reporting
    report_path: Path = Path("validation_report.html")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    potential: PotentialConfig
    lammps: LammpsConfig = Field(default_factory=LammpsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    # New configurations
    structure_gen: StructureGenConfig = Field(discriminator="strategy")
    md: MDConfig

    # Optional components
    dft: Optional[DFTConfig] = None
    training: Optional[TrainingConfig] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io

        data = io.load_yaml(path)
        return cls(**data)
