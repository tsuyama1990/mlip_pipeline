from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.enums import DynamicsType, GeneratorType, OracleType, TrainerType


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Field(..., description="Working directory for the pipeline")
    max_iterations: int = Field(..., gt=0, description="Maximum number of iterations")
    state_file: Path = Field(Path("workflow_state.json"), description="Path to the state file")

class BaseComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(True, description="Whether this component is enabled")

class GeneratorConfig(BaseComponentConfig):
    type: GeneratorType = Field(GeneratorType.MOCK, description="Type of generator")

class OracleConfig(BaseComponentConfig):
    type: OracleType = Field(OracleType.MOCK, description="Type of oracle")

class TrainerConfig(BaseComponentConfig):
    type: TrainerType = Field(TrainerType.MOCK, description="Type of trainer")

class DynamicsConfig(BaseComponentConfig):
    type: DynamicsType = Field(DynamicsType.MOCK, description="Type of dynamics engine")

class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = Field("PyAceMaker", min_length=1, description="Name of the project")
    orchestrator: OrchestratorConfig
    generator: GeneratorConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
