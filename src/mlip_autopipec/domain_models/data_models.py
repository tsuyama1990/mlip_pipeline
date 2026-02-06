
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StructureMetadata(BaseModel):
    """
    Wraps an ASE Atoms object with additional metadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structure: Atoms
    energy: float | None = None
    forces: list[list[float]] | None = None
    virial: list[list[float]] | None = None
    iteration: int = 0

class Dataset(BaseModel):
    """
    A collection of StructureMetadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structures: list[StructureMetadata] = Field(default_factory=list)

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
