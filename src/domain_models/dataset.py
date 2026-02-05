from pydantic import BaseModel, ConfigDict

from .structures import StructureMetadata


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics: dict[str, float]
    passed: bool


class Dataset(BaseModel):
    """
    Abstraction for a collection of labeled structures.
    """

    model_config = ConfigDict(extra="forbid")

    structures: list[StructureMetadata]
    description: str = "Dataset of atomic structures"
