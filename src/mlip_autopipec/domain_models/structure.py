from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from ase import Atoms

class StructureMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    structure: Atoms = Field(..., description="The atomic structure (ASE Atoms object)")
    energy: Optional[float] = Field(None, description="Potential energy of the structure")
    forces: Optional[List[List[float]]] = Field(None, description="Forces acting on atoms")
    virial: Optional[List[List[float]]] = Field(None, description="Virial tensor")
    source: str = Field(..., description="Source of the structure")
    iteration: int = Field(0, description="Active learning cycle iteration number")

class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structures: List[StructureMetadata] = Field(default_factory=list, description="List of structures")
    name: str = Field("dataset", description="Name of the dataset")
