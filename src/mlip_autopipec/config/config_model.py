
from pydantic import BaseModel, ConfigDict, Field, FilePath


class ProjectConfig(BaseModel):
    name: str
    seed: int = 42
    model_config = ConfigDict(extra="forbid")


class ExplorationConfig(BaseModel):
    # Placeholders for future expansion based on SPEC
    md_steps: int = 10000
    model_config = ConfigDict(extra="forbid")


class OracleConfig(BaseModel):
    # Placeholders for future expansion based on SPEC
    scf_accuracy: float = 1e-6
    model_config = ConfigDict(extra="forbid")


class TrainingConfig(BaseModel):
    dataset_path: FilePath
    max_epochs: int = 100
    command: str = "pace_train"
    model_config = ConfigDict(extra="forbid")


class ValidationConfig(BaseModel):
    # Placeholders for future expansion based on SPEC
    run_validation: bool = False
    model_config = ConfigDict(extra="forbid")


class OrchestratorConfig(BaseModel):
    max_iterations: int = 10
    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    project: ProjectConfig
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    training: TrainingConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    model_config = ConfigDict(extra="forbid")
