from pydantic import BaseModel, ConfigDict, Field, FilePath


class ProjectConfig(BaseModel):
    name: str
    seed: int = 42
    model_config = ConfigDict(extra="forbid")


class TrainingConfig(BaseModel):
    dataset_path: FilePath
    max_epochs: int = 100
    command: str = "pace_train"
    model_config = ConfigDict(extra="forbid")


class ExplorationConfig(BaseModel):
    method: str = "md"
    steps: int = 1000
    model_config = ConfigDict(extra="forbid")


class OracleConfig(BaseModel):
    method: str = "dft_mock"
    model_config = ConfigDict(extra="forbid")


class ValidationConfig(BaseModel):
    run_validation: bool = True
    model_config = ConfigDict(extra="forbid")


class OrchestratorConfig(BaseModel):
    max_iterations: int = 10
    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    project: ProjectConfig
    training: TrainingConfig
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    model_config = ConfigDict(extra="forbid")
