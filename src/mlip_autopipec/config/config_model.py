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


class OrchestratorConfig(BaseModel):
    max_iterations: int = 10
    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    project: ProjectConfig
    training: TrainingConfig
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    model_config = ConfigDict(extra="forbid")
