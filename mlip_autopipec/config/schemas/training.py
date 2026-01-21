from pydantic import BaseModel, ConfigDict


class TrainingConfig(BaseModel):
    """Configuration for Pacemaker training."""
    training_data_path: str = "training.p7"
    test_data_path: str = "test.p7"
    model_config = ConfigDict(extra="forbid")
