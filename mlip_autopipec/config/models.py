from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field
from mlip_autopipec.config.schemas.common import TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig

# Try to import other configs from schemas if they exist, else use Any/Dummy
try:
    from mlip_autopipec.config.schemas.training import TrainingConfig as TrainConfig
except ImportError:
    class TrainConfig(BaseModel): pass # type: ignore

try:
    from mlip_autopipec.config.schemas.inference import InferenceConfig
except ImportError:
    class InferenceConfig(BaseModel): pass # type: ignore

try:
    from mlip_autopipec.config.schemas.generator import GeneratorConfig
except ImportError:
    class GeneratorConfig(BaseModel): pass # type: ignore

try:
    from mlip_autopipec.config.schemas.surrogate import SurrogateConfig, EmbeddingConfig
except ImportError:
    class SurrogateConfig(BaseModel): pass # type: ignore
    class EmbeddingConfig(BaseModel): pass # type: ignore

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
    # Mapping to TargetSystem + Resources (approx)
    project_name: str = "MLIP Project"
    target_system: TargetSystem
    # Resources...
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

class SystemConfig(BaseModel):
    """
    Legacy SystemConfig for compatibility with existing modules.
    Wraps MLIPConfig components.
    """
    minimal: Optional[MinimalConfig] = None
    target_system: Optional[TargetSystem] = None
    dft_config: Optional[DFTConfig] = None
    working_dir: Path = Path("_work")
    db_path: Path = Path("mlip.db")
    log_path: Path = Path("mlip.log")

    # Add other fields as Any to satisfy mypy for now
    workflow_config: Any = None
    explorer_config: Any = None
    surrogate_config: Any = None
    training_config: Any = None
    inference_config: Any = None
    generator_config: Any = None
    generator: Any = None
    explorer: Any = None
    dask: Any = None
    dft: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class CheckpointState(BaseModel):
    # Placeholder for workflow state
    run_uuid: Any = None
    system_config: SystemConfig
    active_learning_generation: int = 0
    current_potential_path: Optional[Path] = None
    pending_job_ids: list[Any] = []
    job_submission_args: dict[Any, Any] = {}
    training_history: list[Any] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)
