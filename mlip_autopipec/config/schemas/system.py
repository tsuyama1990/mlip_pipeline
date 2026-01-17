from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import UserInputConfig
from .dft import DFTConfig
from .exploration import ExplorerConfig
from .inference import InferenceConfig
from .training import TrainingConfig, TrainingRunMetrics


class WorkflowConfig(BaseModel):
    checkpoint_filename: str = "checkpoint.json"
    model_config = ConfigDict(extra="forbid")

    @field_validator("checkpoint_filename")
    @classmethod
    def validate_extension(cls, v: str) -> str:
        if not v.endswith(".json"):
            msg = "checkpoint_filename must have a .json extension."
            raise ValueError(msg)
        return v

class SystemConfig(BaseModel):
    """
    Comprehensive configuration for the MLIP-AutoPipe system.
    """
    # Core (Cycle 1)
    user_input: UserInputConfig
    working_dir: Path
    db_path: Path
    log_path: Path
    run_uuid: UUID

    # Workflow (Cycle 8)
    workflow_config: WorkflowConfig = Field(default_factory=WorkflowConfig)

    # Modules (Cycle 2-7)
    dft_config: DFTConfig | None = None
    explorer_config: ExplorerConfig | None = None
    training_config: TrainingConfig | None = None
    inference_config: InferenceConfig | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

class CheckpointState(BaseModel):
    run_uuid: UUID
    system_config: SystemConfig
    active_learning_generation: int = Field(0, ge=0)
    current_potential_path: Path | None = None
    pending_job_ids: list[UUID] = Field(default_factory=list)
    # job_submission_args stores arguments like (atoms,), where atoms is an ase.Atoms object.
    # ase.Atoms is not pydantic-validatable easily, so we allow arbitrary types.
    job_submission_args: dict[UUID, Any] = Field(default_factory=dict)
    training_history: list[TrainingRunMetrics] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

class CalculationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str
    uuid: str
    force_mask: list[list[float]] | None = None
