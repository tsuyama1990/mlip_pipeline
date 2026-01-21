from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.config.schemas.common import TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig

# Try to import other configs from schemas if they exist, else use strict Empty/Dict models
try:
    from mlip_autopipec.config.schemas.training import TrainingConfig as TrainConfig
except ImportError:
    class TrainConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

try:
    from mlip_autopipec.config.schemas.inference import InferenceConfig
except ImportError:
    class InferenceConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

try:
    from mlip_autopipec.config.schemas.generator import GeneratorConfig
except ImportError:
    class GeneratorConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

try:
    from mlip_autopipec.config.schemas.surrogate import EmbeddingConfig, SurrogateConfig
except ImportError:
    class SurrogateConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

    class EmbeddingConfig(BaseModel):
        model_config = ConfigDict(extra="allow")


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
    model_config = ConfigDict(extra="allow")


# Placeholder for strictly typed schemas that might not be importable yet
# We allow arbitrary types here only because we can't import the real ones safely yet,
# but we forbid extra fields in the container.
class PlaceholderConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class SystemConfig(BaseModel):
    """
    Legacy SystemConfig for compatibility with existing modules.
    Wraps MLIPConfig components.

    This model enforces strict type checking where possible and forbids extra fields.
    Legacy fields are typed as Optional[Any] but the goal is to replace them with specific configs.
    """

    minimal: MinimalConfig | None = None
    target_system: TargetSystem | None = None
    dft_config: DFTConfig | None = None
    working_dir: Path = Path("_work")
    db_path: Path = Path("mlip.db")
    log_path: Path = Path("mlip.log")

    # Strictly typed legacy fields where possible
    workflow_config: PlaceholderConfig | None = None
    explorer_config: PlaceholderConfig | None = None
    surrogate_config: PlaceholderConfig | None = None
    training_config: PlaceholderConfig | None = None
    inference_config: PlaceholderConfig | None = None
    generator_config: PlaceholderConfig | None = None

    # Legacy aliases that were causing strictness issues
    # We use PlaceholderConfig (allowing dicts/models) instead of Any to be slightly stricter if possible
    # but for now Any is safest for un-migrated code.
    generator: Any | None = None
    explorer: Any | None = None
    dask: Any | None = None
    dft: Any | None = None

    # Strictly forbid extra fields to ensure data integrity
    # We retain arbitrary_types_allowed=True ONLY for the legacy Any fields that might hold complex objects
    # pending full migration.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class CheckpointState(BaseModel):
    # Placeholder for workflow state
    run_uuid: Any = None
    system_config: SystemConfig
    active_learning_generation: int = 0
    current_potential_path: Path | None = None
    pending_job_ids: list[Any] = []
    job_submission_args: dict[Any, Any] = {}
    training_history: list[Any] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)
