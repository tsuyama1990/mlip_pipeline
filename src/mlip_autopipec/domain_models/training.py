from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from mlip_autopipec import defaults


class TrainingConfig(BaseModel):
    """Configuration for Pacemaker training."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = defaults.DEFAULT_TRAIN_BATCH
    max_epochs: int = defaults.DEFAULT_TRAIN_EPOCHS

    # Optimizer settings
    ladder_step: List[int] = Field(default_factory=lambda: defaults.DEFAULT_TRAIN_LADDER)
    kappa: float = defaults.DEFAULT_TRAIN_KAPPA

    # Active Set
    active_set_optimization: bool = defaults.DEFAULT_ACTIVE_SET_OPT
    max_active_set_size: int = defaults.DEFAULT_MAX_ACTIVE_SET

    # Potential initialization
    initial_potential: Optional[Path] = None

    # Dataset and Output
    dataset_path: Optional[Path] = None
    work_dir: Path = Path(defaults.DEFAULT_TRAIN_WORK_DIR)

    @field_validator("batch_size", "max_epochs", "max_active_set_size")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class TrainingResult(BaseModel):
    """Result of a training run."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str  # JobStatus enum as string
    work_dir: Path
    duration_seconds: float
    log_content: str
    potential_path: Optional[Path] = None
    validation_metrics: dict[str, float] = Field(default_factory=dict)
