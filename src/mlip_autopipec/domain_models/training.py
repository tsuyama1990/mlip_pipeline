from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

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


class TrainingResult(JobResult):
    """
    Result of a Potential Training job.
    """

    model_config = ConfigDict(extra="forbid")

    potential_path: Path
    validation_metrics: dict[str, float]
