from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .dft import DFTConfig
from .exploration import ExplorerConfig, ExplorerParams, GeneratorParams
from .inference import InferenceConfig
from .training import TrainingConfig


class WorkflowConfig(BaseModel):
    checkpoint_filename: str = "checkpoint.json"
    model_config = ConfigDict(extra="forbid")

class DaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheduler_address: str | None = Field(None)

class SystemConfig(BaseModel):
    project_name: str
    run_uuid: UUID
    workflow_config: WorkflowConfig = Field(default_factory=WorkflowConfig)

    # Modules
    dft_config: DFTConfig
    explorer_config: ExplorerConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig

    # Optional components (from previous version compat)
    generator: GeneratorParams | None = None
    explorer: ExplorerParams | None = None
    dask: DaskConfig | None = None
    db_path: str = "mlip_database.db"

    model_config = ConfigDict(extra="forbid")

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        import os
        if ".." in v or os.path.isabs(v):
            raise ValueError("db_path must be a relative path.")
        return v

class CheckpointState(BaseModel):
    run_uuid: UUID
    system_config: SystemConfig
    active_learning_generation: int = Field(0, ge=0)
    current_potential_path: Path | None = None
    pending_job_ids: list[UUID] = Field(default_factory=list)
    job_submission_args: dict[UUID, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

class CalculationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str
    uuid: str
    force_mask: list[list[float]] | None = None
