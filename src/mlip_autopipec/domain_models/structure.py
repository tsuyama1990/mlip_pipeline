from typing import Any

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class Structure(BaseModel):
    """
    Domain model representing an atomic structure with metadata.
    """
    atoms: Atoms = Field(description="The ASE Atoms object representing the structure")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the structure")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

class Dataset(BaseModel):
    """
    Domain model representing a collection of labeled structures.
    """
    structures: list[Structure] = Field(default_factory=list, description="List of Structure objects")
    name: str = Field(default="dataset", description="Name of the dataset")

    def __len__(self) -> int:
        return len(self.structures)

    def to_atoms_list(self) -> list[Atoms]:
        return [s.atoms for s in self.structures]

    model_config = ConfigDict(extra="forbid")
