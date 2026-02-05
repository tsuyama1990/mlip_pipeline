from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    """
    Data model for an atomic structure with associated metadata.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    structure: Atoms
    source: str = Field(..., description="Source of the structure (e.g., 'random', 'seed')")
    generation_method: str = Field(..., description="Method used to generate the structure")
