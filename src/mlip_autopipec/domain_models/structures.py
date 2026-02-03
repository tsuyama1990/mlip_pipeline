from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    """Metadata tracking the genealogy and status of an atomic structure."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="Source of the structure (e.g., 'initial', 'exploration')")
    generation_method: str = Field(..., description="Method used to generate the structure")
    filepath: str = Field(..., description="Path to the structure file")
    parent_structure_id: str | None = Field(None, description="ID of the parent structure")
    uncertainty: float | None = Field(None, description="Uncertainty metric (e.g., max gamma)")
    selection_score: float | None = Field(None, description="Score used for active set selection")
