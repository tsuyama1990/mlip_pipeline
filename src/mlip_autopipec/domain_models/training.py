from pathlib import Path
from typing import Optional, Any

from pydantic import BaseModel, ConfigDict, field_validator

from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.domain_models.potential import Potential


class TrainingConfig(BaseModel):
    """Configuration for Training (Pacemaker)."""

    model_config = ConfigDict(extra="forbid")

    initial_potential: Optional[Path] = None
    active_set_optimization: bool = True
    max_epochs: int = 100
    batch_size: int = 100
    ladder_step: list[int] = [10, 5]
    kappa: float = 0.3

    @field_validator("batch_size", "max_epochs")
    @classmethod
    def validate_positive_ints(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("kappa")
    @classmethod
    def validate_kappa(cls, v: float) -> float:
        if v < 0:
             raise ValueError("kappa must be non-negative")
        return v


class TrainingResult(JobResult):
    """
    Result of a Potential Training job.
    """

    model_config = ConfigDict(extra="forbid")

    potential: Potential
    validation_metrics: dict[str, float]
