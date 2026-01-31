from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.job import JobResult


class TrainingConfig(BaseModel):
    """Configuration for Training (Pacemaker)."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = 100
    max_epochs: int = 100
    ladder_step: list[int] = Field(default_factory=lambda: [100, 10])
    kappa: float = 0.4
    initial_potential: Optional[Path] = None
    active_set_optimization: bool = True


class TrainingResult(JobResult):
    """
    Result of a Potential Training job.
    """

    model_config = ConfigDict(extra="forbid")

    potential_path: Path
    validation_metrics: dict[str, float]
