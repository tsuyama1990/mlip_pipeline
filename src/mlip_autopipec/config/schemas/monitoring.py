from pydantic import BaseModel, ConfigDict, RootModel

from .training import TrainingMetrics


class DatasetComposition(RootModel[dict[str, int]]):
    # RootModel does not support extra="forbid"
    pass


class DashboardData(BaseModel):
    project_name: str
    current_generation: int
    completed_calcs: int
    pending_calcs: int
    training_history: list[TrainingMetrics]
    dataset_composition: DatasetComposition
    model_config = ConfigDict(extra="forbid")
