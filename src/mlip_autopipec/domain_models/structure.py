
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    structure: Atoms = Field(..., description="The atomic structure (ASE Atoms object)")
    energy: float | None = Field(default=None, description="Potential energy of the structure")
    forces: list[list[float]] | None = Field(default=None, description="Forces acting on atoms")
    virial: list[list[float]] | None = Field(default=None, description="Virial tensor")
    source: str = Field(..., description="Source of the structure")
    iteration: int = Field(default=0, description="Active learning cycle iteration number")

class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structures: list[StructureMetadata] = Field(default_factory=list, description="List of structures")
    name: str = Field(default="dataset", description="Name of the dataset")
