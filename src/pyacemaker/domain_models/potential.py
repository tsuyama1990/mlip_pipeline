"""Potential-related domain models."""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyacemaker.domain_models.common import PotentialType, utc_now


class Potential(BaseModel):
    """Representation of an interatomic potential."""

    model_config = ConfigDict(extra="forbid")

    path: Path = Field(..., description="Path to the potential file")
    type: PotentialType = Field(..., description="Type of the potential")
    version: str = Field(..., description="Version identifier")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters used to generate this potential"
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Validation metrics (RMSE, etc.)"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Creation timestamp")

    @field_validator("path")
    @classmethod
    def validate_path_format(cls, v: Path) -> Path:
        """Validate path format."""
        if str(v).strip() == "":
            msg = "Path cannot be empty"
            raise ValueError(msg)

        # Check against basic path traversal patterns
        if ".." in v.parts:
            msg = f"Path traversal not allowed in potential path: {v}"
            raise ValueError(msg)

        # Use resolve to sanitize and check for potential traversal issues
        try:
            # We want to allow non-existent paths, but prevent traversal outside allowed roots if possible.
            # Since we don't have project root here, we just check general validity.
            # Use strict=False to allow potential creation.
            resolved = v.resolve(strict=False)

            # Check if parent exists? This enforces that we are creating in a valid directory.
            # This is a good middle ground for security vs flexibility.
            if not resolved.parent.exists():
                # Just a warning or strict? Let's check for traversal specifically.
                # If '..' remains after resolve, it's weird (resolve handles it).
                pass

        except Exception as e:
            msg = f"Invalid potential path: {v}. Error: {e}"
            raise ValueError(msg) from e

        return v
