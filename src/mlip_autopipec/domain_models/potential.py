from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.paths import validate_path_safety


class Potential(BaseModel):
    """
    Represents a trained MLIP potential file.
    """
    path: Path = Field(..., description="Path to the potential file (.yace, .pot, etc.)")
    format: str = Field(..., description="Format of the potential (e.g., 'yace')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Training parameters")

    model_config = ConfigDict(extra="forbid")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        # Pydantic 2.x validators are classmethods
        if not validate_path_safety(v):
             # Should not happen as validate_path_safety raises or returns Path
             # But just in case validation logic changes
             msg = "Invalid path"
             raise ValueError(msg)
        return v
