from typing import Any

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class Structure(BaseModel):
    """
    Represents an atomic configuration with metadata.
    Wraps ase.Atoms but adds tracking information.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    atoms: Atoms
    energy: float | None = None
    forces: list[list[float]] | None = None
    stress: list[float] | None = None

    # Metadata
    source: str = Field(..., description="Source of the structure (e.g., MD_HALT, RANDOM)")
    cycle: int = Field(..., description="Active Learning Cycle number")
    label: str | None = Field(None, description="Unique identifier or tag")
    tags: dict[str, Any] = Field(default_factory=dict, description="Additional metadata tags")

    def get_ase_atoms(self) -> Atoms:
        """Returns the underlying ASE Atoms object."""
        return self.atoms
