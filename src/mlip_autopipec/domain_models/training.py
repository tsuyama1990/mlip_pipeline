from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrainingConfig(BaseModel):
    """Configuration for Pacemaker training."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = 100
    max_epochs: int = 100

    # Optimizer settings
    ladder_step: List[int] = Field(default_factory=lambda: [100, 10])
    kappa: float = 0.4

    # Active Set
    active_set_optimization: bool = True
    max_active_set_size: int = 1000

    # Potential initialization
    initial_potential: Optional[Path] = None

    # Dataset and Output
    dataset_path: Optional[Path] = None
    work_dir: Path = Path("training_work")

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
