from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    """
    Metadata for an atomic structure, including provenance and the structure itself.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    # The actual atomic structure (ASE Atoms object)
    structure: Atoms = Field(..., description="The ASE Atoms object representing the structure.")

    # Provenance information
    source: str = Field(..., description="The source of the structure (e.g., 'md', 'random', 'seed').")
    generation_method: str = Field(..., description="The specific method used to generate this structure.")
    parent_structure_id: str | None = Field(None, description="ID of the parent structure, if applicable.")

    # Validation/Selection metrics
    uncertainty: float | None = Field(None, description="Uncertainty score associated with the structure.")
    selection_score: float | None = Field(None, description="Score used for active set selection.")

    # File storage info (optional for now, but good for tracking)
    filepath: str | None = Field(None, description="Path to the saved structure file (e.g., .xyz, .cif).")

    def __repr__(self) -> str:
        return f"<StructureMetadata(source={self.source}, method={self.generation_method}, atoms={len(self.structure)})>"
