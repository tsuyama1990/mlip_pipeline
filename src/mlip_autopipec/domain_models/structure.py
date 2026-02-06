from pathlib import Path

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    @field_validator("structure")
    @classmethod
    def check_structure_not_none(cls, v: Atoms) -> Atoms:
        if v is None:
            msg = "Structure cannot be None"
            raise ValueError(msg)
        return v


class Dataset(BaseModel):
    """
    A collection of StructureMetadata.
    Can handle in-memory structures OR a reference to a file.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structures: list[StructureMetadata] = Field(default_factory=list)
    file_path: Path | None = None

    @model_validator(mode="after")
    def check_mutually_exclusive(self) -> "Dataset":
        if self.file_path is not None and len(self.structures) > 0:
            msg = "Dataset cannot have both 'file_path' and non-empty 'structures'."
            raise ValueError(msg)
        return self
