
from pydantic import BaseModel, ConfigDict, Field, FilePath


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Name of the project.")
    seed: int = Field(42, description="Random seed for reproducibility.")

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_path: FilePath = Field(..., description="Path to the training dataset (.pckl, .xyz).")
    max_epochs: int = Field(100, ge=1, description="Maximum number of training epochs.")
    command: str = Field("pace_train", description="Command to execute Pacemaker training.")

class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_iterations: int = Field(10, ge=1, description="Maximum number of active learning cycles.")

class Config(BaseModel):
    """
    Root configuration object for PYACEMAKER.
    """
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    training: TrainingConfig
    orchestrator: OrchestratorConfig = Field(default_factory=lambda: OrchestratorConfig(max_iterations=10))
