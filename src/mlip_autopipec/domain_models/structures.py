from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    """
    Metadata wrapper for atomic structures to track provenance and properties.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    atoms: Atoms = Field(..., description="The atomic structure object")
    source: str = Field(..., description="Source of the structure (e.g., 'random', 'md_snapshot')")
    generation_method: str = Field(..., description="Method used to generate it")
    parent_structure_id: str | None = Field(None, description="ID of parent structure if any")
    uncertainty: float | None = Field(None, description="Uncertainty score of this structure")
    filepath: str | None = Field(None, description="Path to saved file")
    selection_score: float | None = Field(None, description="Active learning selection score")
