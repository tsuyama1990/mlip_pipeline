from pydantic import BaseModel, ConfigDict, Field


class ProductionManifest(BaseModel):
    version: str
    author: str
    training_set_size: int
    validation_metrics: dict[str, float | bool | str]
    creation_date: str = Field(description="ISO 8601 timestamp")

    model_config = ConfigDict(extra="forbid")
