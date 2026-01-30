from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from mlip_autopipec.domain_models.job import JobResult


class TrainingConfig(BaseModel):
    """Configuration for Training (Pacemaker)."""

    model_config = ConfigDict(extra="forbid")
    initial_potential: Optional[Path] = None
    active_set_optimization: bool = True
    max_epochs: int = 100


class TrainingResult(JobResult):
    """
    Result of a Potential Training job.
    """

    model_config = ConfigDict(extra="forbid")

    potential_path: Path
    validation_metrics: dict[str, float]
