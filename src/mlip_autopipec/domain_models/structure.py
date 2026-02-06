from pathlib import Path

from ase import Atoms
from pydantic import BaseModel, ConfigDict, field_validator


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
    A reference to a file containing structures (e.g., .xyz, .extxyz).
    Strictly file-based to ensure scalability and prevent OOM errors.
    """

    model_config = ConfigDict(extra="forbid")

    file_path: Path
