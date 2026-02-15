"""Validator domain models."""


from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    """Result of a validation run."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(..., description="Whether validation passed")
    metrics: dict[str, float] = Field(..., description="Validation metrics")
    phonon_stable: bool = Field(..., description="Whether phonon stability check passed")
    elastic_stable: bool = Field(..., description="Whether elastic stability check passed")
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="Paths to generated artifacts (e.g. plots)"
    )
