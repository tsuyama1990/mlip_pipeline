from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FilePath


class DFTConfig(BaseModel):
    command: str = "pw.x"
    pseudopotentials: dict[str, str]  # Element -> Filename
    kspacing: float = 0.04  # Inverse distance for K-grid
    ecutwfc: float = 50.0  # Wavefunction cutoff (Ry)
    max_retries: int = 3
    model_config = ConfigDict(extra="forbid")


class LammpsConfig(BaseModel):
    command: str = "lmp"
    num_processors: int = 1
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(BaseModel):
    name: str
    seed: int = 42
    model_config = ConfigDict(extra="forbid")


class TrainingConfig(BaseModel):
    dataset_path: FilePath
    max_epochs: int = 100
    command: str = "pace_train"
    model_config = ConfigDict(extra="forbid")


class OrchestratorConfig(BaseModel):
    max_iterations: int = 10
    model_config = ConfigDict(extra="forbid")


class StructureGenConfig(BaseModel):
    strategy: str = "adaptive"
    parameters: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class OracleConfig(BaseModel):
    method: str = "dft"
    model_config = ConfigDict(extra="forbid")


class ValidationConfig(BaseModel):
    run_validation: bool = True
    check_phonons: bool = True
    check_elastic: bool = True
    model_config = ConfigDict(extra="forbid")


class SelectionConfig(BaseModel):
    method: str = "random"
    max_structures: int = 100
    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    project: ProjectConfig
    training: TrainingConfig
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    exploration: StructureGenConfig = Field(default_factory=StructureGenConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    dft: DFTConfig | None = None
    lammps: LammpsConfig | None = None

    model_config = ConfigDict(extra="forbid")
