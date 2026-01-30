from pathlib import Path
from typing import Optional, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.job import JobResult


class TrainingConfig(BaseModel):
    """Configuration for Training (Pacemaker)."""

    model_config = ConfigDict(extra="forbid")

    # Core Pacemaker parameters
    batch_size: int = 100
    max_epochs: int = 1000
    ladder_step: list[int] = Field(default_factory=lambda: [10, 50, 100])
    kappa: float = 0.5  # Weighting of forces vs energy

    # Workflow parameters
    initial_potential: Optional[Path] = None
    active_set_selection: bool = True  # Renamed from active_set_optimization to match UAT/concept

    @field_validator("batch_size", "max_epochs")
    @classmethod
    def validate_positive(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return v

    @field_validator("kappa")
    @classmethod
    def validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("kappa must be between 0.0 and 1.0")
        return v

    @field_validator("ladder_step")
    @classmethod
    def validate_ladder_step(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("ladder_step must not be empty")
        if any(x <= 0 for x in v):
            raise ValueError("ladder_step elements must be greater than 0")
        return v


class TrainingResult(JobResult):
    """
    Result of a Potential Training job.
    """

    model_config = ConfigDict(extra="forbid")

    potential_path: Path
    validation_metrics: dict[str, float]
