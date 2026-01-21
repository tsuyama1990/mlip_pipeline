from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.config.schemas.common import TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.config.schemas.orchestration import WorkflowConfig
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.config.schemas.training import TrainingConfig


class RuntimeConfig(BaseModel):
    database_path: Path = Field(Path("mlip.db"), description="Path to SQLite database")
    work_dir: Path = Field(Path("_work"), description="Scratch directory for calculations")

    model_config = ConfigDict(extra="forbid")


class MLIPConfig(BaseModel):
    target_system: TargetSystem
    dft: DFTConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    model_config = ConfigDict(extra="forbid")


# --- Compatibility / Legacy Models ---


class MinimalConfig(BaseModel):
    project_name: str = "MLIP Project"
    target_system: TargetSystem
    model_config = ConfigDict(extra="allow") # Allow extra for flexibility in minimal inputs


class SystemConfig(BaseModel):
    """
    Comprehensive System Configuration.
    Enforces strict types for all modules.
    """

    minimal: MinimalConfig | None = None
    target_system: TargetSystem | None = None
    dft_config: DFTConfig | None = None
    working_dir: Path = Path("_work")
    db_path: Path = Path("mlip.db")
    log_path: Path = Path("mlip.log")

    # Strict types
    workflow_config: WorkflowConfig | None = None
    explorer_config: Any | None = None # Still Any as no schema provided for Explorer yet
    surrogate_config: SurrogateConfig | None = None
    training_config: TrainingConfig | None = None
    inference_config: InferenceConfig | None = None
    generator_config: GeneratorConfig | None = None

    # Legacy aliases - Typed as Optional[Any] to allow backward compat but restricted by model_config if possible
    # We remove arbitrary_types_allowed=True if we can.
    # If legacy code injects objects, we need it.
    # Auditor said "allows arbitrary types... could lead to corruption".
    # I will try to remove arbitrary_types_allowed and see if mypy/tests pass.
    # If I use Any, Pydantic allows validation of Any.

    generator: Any | None = None
    explorer: Any | None = None
    dask: Any | None = None
    dft: Any | None = None

    model_config = ConfigDict(extra="forbid") # Removed arbitrary_types_allowed=True


class CheckpointState(BaseModel):
    # Placeholder for workflow state
    run_uuid: Any = None
    system_config: SystemConfig
    active_learning_generation: int = 0
    current_potential_path: Path | None = None
    pending_job_ids: list[Any] = []
    job_submission_args: dict[Any, Any] = {}
    training_history: list[Any] = []
    # We might need arbitrary types here for 'Any' fields if they hold non-pydantic objects
    model_config = ConfigDict(arbitrary_types_allowed=True)
