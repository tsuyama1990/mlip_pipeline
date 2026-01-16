
from pydantic import BaseModel, ConfigDict

from .training import TrainingRunMetrics


class DashboardData(BaseModel):
    project_name: str
    current_generation: int
    completed_calcs: int
    pending_calcs: int
    training_history: list[TrainingRunMetrics]
    dataset_composition: dict[str, int]
    model_config = ConfigDict(extra="forbid")
