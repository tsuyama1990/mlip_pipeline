from pydantic import BaseModel, ConfigDict, field_validator


class ValidationResult(BaseModel):
    """
    Results from a validation run.
    """
    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float]
    is_stable: bool

    @field_validator("metrics")
    @classmethod
    def check_metrics_not_empty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            msg = "Metrics dictionary cannot be empty"
            raise ValueError(msg)
        return v

    def __str__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        status = "Stable" if self.is_stable else "Unstable"
        return f"ValidationResult(Status: {status}, Metrics: {{{metrics_str}}})"
