from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool = Field(..., description="Whether the validation passed")
    metrics: dict[str, float] = Field(default_factory=dict, description="Key metrics like RMSE")
    artifacts: dict[str, Path] = Field(default_factory=dict, description="Paths to generated artifacts")
