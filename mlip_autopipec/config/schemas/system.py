from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import Resources, SimulationGoal, TargetSystem
from .dft import DFTConfig
from .exploration import ExplorerConfig, ExplorerParams, GeneratorParams
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


class DaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheduler_address: str | None = Field(None)


class SystemConfig(BaseModel):
    """
    Comprehensive configuration for the MLIP-AutoPipe system.
    Most fields are optional to allow for incremental configuration or testing of specific modules,
    but in a full production run, the relevant sections must be present.
    """

    project_name: str
    run_uuid: UUID
    workflow_config: WorkflowConfig = Field(default_factory=WorkflowConfig)

    # Core System Definition (Optional primarily for component-level testing)
    target_system: TargetSystem | None = None
    resources: Resources | None = None
    simulation_goal: SimulationGoal | None = None

    # Module Configurations
    dft_config: DFTConfig | None = None
    explorer_config: ExplorerConfig | None = None
    training_config: TrainingConfig | None = None
    inference_config: InferenceConfig | None = None

    # Legacy / Transition Fields (Deprecated)
    generator: GeneratorParams | None = None
    explorer: ExplorerParams | None = None
    dask: DaskConfig | None = None
    dft: DFTConfig | None = Field(
        None,
        description="Legacy alias for `dft_config`. Please use `dft_config` instead. This field is deprecated and will be removed in future versions.",
    )

    db_path: str = "mlip_database.db"
    working_dir: Path | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        import os

        if ".." in v or os.path.isabs(v):
            msg = "db_path must be a relative path."
            raise ValueError(msg)
        return v


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
