from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExtractedStructure(BaseModel):
    """
    Data model for an extracted local environment (cluster-in-box).

    Attributes:
        atoms: The ASE Atoms object representing the extracted cluster.
        origin_uuid: The UUID of the original MD frame.
        origin_index: The index of the focal atom in the original frame.
        mask_radius: The radius used for force masking (core radius).
    """
    atoms: Any = Field(..., description="The ASE Atoms object")
    origin_uuid: str = Field(..., description="UUID of the original MD frame")
    origin_index: int = Field(..., description="Index of the focal atom in original frame")
    mask_radius: float = Field(..., description="Radius used for force masking")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
