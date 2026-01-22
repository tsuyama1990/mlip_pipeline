from pydantic import BaseModel, ConfigDict


class TrainConfig(BaseModel):
    """Configuration for Pacemaker training."""
    training_data_path: str = "training.p7"
    test_data_path: str = "test.p7"
    model_config = ConfigDict(extra="forbid")

class TrainingConfig(TrainConfig):
    """Alias for TrainConfig to satisfy imports."""

class TrainingResult(BaseModel):
    """Result of training."""
    success: bool
    metrics: dict[str, float]

class TrainingData(BaseModel):
    """Placeholder for training data model."""
