from pydantic import BaseModel, ConfigDict, Field

from .structures import StructureMetadata


class Dataset(BaseModel):
    """
    Abstraction for a collection of labeled structures.
    """

    model_config = ConfigDict(extra="forbid")

    structures: list[StructureMetadata] = Field(default_factory=list)
