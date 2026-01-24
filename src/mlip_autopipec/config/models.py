from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.config.schemas.core import RuntimeConfig, TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.config.schemas.exploration import ExplorerConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.config.schemas.inference import InferenceConfig
from mlip_autopipec.config.schemas.surrogate import SurrogateConfig
from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.config.schemas.validation import ValidationConfig
from mlip_autopipec.config.schemas.workflow import WorkflowConfig


class MLIPConfig(BaseModel):
    """
    Root configuration object for MLIP-AutoPipe.
    Aggregates all module configurations.
    """

    target_system: TargetSystem
    dft: DFTConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generator_config: GeneratorConfig = Field(default_factory=GeneratorConfig)
    surrogate_config: SurrogateConfig = Field(default_factory=SurrogateConfig)
    training_config: TrainingConfig | None = None
    inference_config: InferenceConfig | None = None
    validation_config: ValidationConfig | None = None

    # Added for Orchestration
    workflow_config: WorkflowConfig | None = Field(None, alias="workflow")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --- Compatibility / Legacy Models ---


class MinimalConfig(BaseModel):
    project_name: str = "MLIP Project"
    target_system: TargetSystem
    model_config = ConfigDict(extra="forbid")


class SystemConfig(BaseModel):
    """
    Comprehensive System Configuration.
    Enforces strict types for all modules.

    This model serves as the central state object for configuration throughout the lifecycle.
    """

    minimal: MinimalConfig | None = None
    target_system: TargetSystem | None = None
    dft_config: DFTConfig | None = None

    # Use defaults but allow override.
    # TODO: In future iterations, move these defaults to a global constants file.
    working_dir: Path = Path("_work")
    db_path: Path = Path("mlip.db")
    log_path: Path = Path("mlip.log")

    # Strict types
    workflow_config: WorkflowConfig | None = None
    explorer_config: ExplorerConfig | None = None
    surrogate_config: SurrogateConfig | None = None
    training_config: TrainingConfig | None = None
    inference_config: InferenceConfig | None = None
    validation_config: ValidationConfig | None = None
    generator_config: GeneratorConfig = Field(default_factory=GeneratorConfig)

    # Legacy aliases - Typed as specific configs where possible, but kept optional for backward compat
    # We remove 'Any' usage.

    generator: GeneratorConfig | None = None
    explorer: ExplorerConfig | None = None
    # dask: Removed to enforce usage of workflow_config
    dft: DFTConfig | None = None

    model_config = ConfigDict(extra="forbid")


class CheckpointState(BaseModel):
    # Placeholder for workflow state
    run_uuid: str | None = None
    system_config: SystemConfig
    active_learning_generation: int = 0
    current_potential_path: Path | None = None
    pending_job_ids: list[str] = []
    job_submission_args: dict[str, Any] = {}
    training_history: list[dict[str, Any]] = []

    model_config = ConfigDict(extra="forbid")
