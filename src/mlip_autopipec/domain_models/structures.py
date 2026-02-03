
from pydantic import BaseModel, ConfigDict, Field


class StructureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="Source of the structure (e.g. 'md', 'random')")
    generation_method: str = Field(..., description="Method used to generate")
    parent_structure_id: str | None = Field(None, description="ID of the parent structure")
    selection_score: float | None = Field(None, description="Score used for selection")
    uncertainty: float | None = Field(None, description="Uncertainty metric")
