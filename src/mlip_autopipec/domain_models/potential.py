from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PotentialType = Literal["ace", "mnn", "mock"]


class Potential(BaseModel):
    """
    Represents a trained machine learning potential.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the potential")
    potential_type: PotentialType = Field(..., description="Type of potential")
    version: str = Field(..., description="Version identifier")
    path: str = Field(..., description="Path to the potential file")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
