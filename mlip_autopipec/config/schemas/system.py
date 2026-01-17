from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import TargetSystem, MinimalConfig
from .dft import DFTConfig
from .exploration import ExplorerConfig, ExplorerParams, GeneratorParams
from .inference import InferenceConfig
from .training import TrainingConfig, TrainingRunMetrics


class WorkflowConfig(BaseModel):
    checkpoint_filename: str = "checkpoint.json"
    model_config = ConfigDict(extra="forbid")

class DaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheduler_address: str | None = Field(None)

class SystemConfig(BaseModel):
    """
    Comprehensive configuration for the MLIP-AutoPipe system.
    Cycle 1 Definition.
    """
    minimal: MinimalConfig
    working_dir: Path
    db_path: Path
    log_path: Path

    # Fields from previous implementation kept as Optional/None for compatibility during refactor
    # They are not part of Cycle 1 SPEC strictly but might be used by existing WorkflowManager tests.
    # We make them optional defaults.
    project_name: str | None = None
    run_uuid: UUID | None = None
    workflow_config: WorkflowConfig | None = None
    target_system: TargetSystem | None = None
    dft_config: DFTConfig | None = None
    explorer_config: ExplorerConfig | None = None
    training_config: TrainingConfig | None = None
    inference_config: InferenceConfig | None = None

    # Legacy / Transition Fields (Deprecated)
    generator: GeneratorParams | None = None
    explorer: ExplorerParams | None = None
    dask: DaskConfig | None = None
    dft: DFTConfig | None = None # Legacy alias

    model_config = ConfigDict(frozen=True, extra="ignore")

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
