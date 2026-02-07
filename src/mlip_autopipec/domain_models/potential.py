from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mlip_autopipec.domain_models.structure import Structure


class Potential(BaseModel):
    """
    Represents a trained potential model.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    version: str
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        if not v.exists():
            msg = f"Path {v} does not exist."
            raise ValueError(msg)
        if not v.is_file():
            msg = f"Path {v} is not a file."
            raise ValueError(msg)
        return v


class ExplorationResult(BaseModel):
    """
    Result of a dynamics exploration run.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["halted", "converged", "error"]
    structure: Structure | None = None
    trajectory: list[Structure] | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_structure_presence(self) -> "ExplorationResult":
        if self.status == "halted" and self.structure is None:
            msg = "Structure must be provided when status is 'halted'."
            raise ValueError(msg)
        return self
