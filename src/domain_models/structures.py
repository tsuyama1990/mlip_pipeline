from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    """
    Metadata wrapper for an atomic structure (ase.Atoms).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    structure: Atoms = Field(..., description="The atomic structure")
    source: str = Field(
        ..., min_length=1, description="Source of the structure (e.g., 'random', 'md')"
    )
    generation_method: str = Field(
        ..., min_length=1, description="Method used to generate the structure"
    )
