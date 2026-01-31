from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.job import JobResult


class TrainingConfig(BaseModel):
    """Configuration for Training (Pacemaker)."""

    model_config = ConfigDict(extra="forbid")
    initial_potential: Optional[Path] = None
    active_set_optimization: bool = True

    # New fields
    batch_size: int = 100
    max_epochs: int = 100
    ladder_step: list[int] = Field(default_factory=lambda: [100, 10])
    kappa: float = 0.3

    @field_validator("batch_size", "max_epochs")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            msg = "Value must be positive"
            raise ValueError(msg)
        return v

    @field_validator("kappa")
    @classmethod
    def validate_kappa(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            msg = "Kappa must be between 0 and 1"
            raise ValueError(msg)
        return v


class TrainingResult(JobResult):
    """
    Result of a Potential Training job.
    """

    model_config = ConfigDict(extra="forbid")

    potential_path: Path
    validation_metrics: dict[str, float]
