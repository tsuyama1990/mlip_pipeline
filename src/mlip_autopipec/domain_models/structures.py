from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    """Metadata for an atomic structure."""
    model_config = ConfigDict(extra='forbid')

    source: str = Field(..., description="Source of the structure (e.g., 'md', 'random', 'user')")
    generation_method: str = Field(..., description="Method used to generate the structure")
    parent_structure_id: str | None = Field(None, description="ID of the parent structure if any")
    selection_score: float | None = Field(None, description="Score used for active learning selection")
    uncertainty: float | None = Field(None, description="Uncertainty metric if calculated")
