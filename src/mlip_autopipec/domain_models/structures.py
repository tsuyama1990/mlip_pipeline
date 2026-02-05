from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    """
    Domain model representing an atomic structure with associated metadata and provenance.
    Wraps an ase.Atoms object.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    structure: Atoms = Field(..., description="The ASE Atoms object")
    id: str = Field(..., description="Unique identifier for the structure")
    source: str = Field(
        ..., description="Source of the structure (e.g. 'initial', 'md', 'relaxation')"
    )
    generation_method: str = Field(..., description="Method used to generate this structure")
    filepath: str | None = Field(None, description="Path to the structure file if saved to disk")
    parent_structure_id: str | None = Field(
        None, description="ID of the parent structure if applicable"
    )
    uncertainty: float | None = Field(None, description="Uncertainty metric for this structure")
    selection_score: float | None = Field(
        None, description="Score used for active learning selection"
    )
