"""Potential-related domain models."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyacemaker.core.config import CONSTANTS
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

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format (semver-like)."""
        if not v:
            msg = "Version string cannot be empty"
            raise ValueError(msg)
        # Basic semver check or alphanumeric
        if not re.match(r"^v?\d+\.\d+(\.\d+)?(-[\w\.]+)?$", v) and (not v.isalnum() or "." not in v):
             # Allow simpler versions for now but warn/restrict
             msg = f"Invalid version format: {v}"
             raise ValueError(msg)
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters based on potential type is hard here without accessing 'type' field.
        We can check general sanity."""
        if not isinstance(v, dict):
            msg = "Parameters must be a dictionary"
            raise TypeError(msg)
        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate metrics."""
        for name, value in v.items():
            if value < 0 and "rmse" in name.lower():
                 msg = f"Metric {name} must be non-negative"
                 raise ValueError(msg)
        return v

    @field_validator("path")
    @classmethod
    def validate_path_format(cls, v: Path) -> Path:
        """Validate path format with strict security checks."""
        if str(v).strip() == "":
            msg = "Path cannot be empty"
            raise ValueError(msg)

        if CONSTANTS.skip_file_checks:
            return v

        try:
            import os
            cwd = Path.cwd().resolve(strict=True)

            # 1. Symlink check (Race Condition Prevention)
            # If file exists, check if it's a symlink directly before resolution
            # If it doesn't exist, we check parent components
            # Note: Checking lstat on the path itself if it exists
            if v.exists() and v.is_symlink():
                 msg = f"Potential path cannot be a symlink: {v}"
                 raise ValueError(msg)

            # 2. Atomic Resolution (Realpath)
            # os.path.realpath resolves symlinks and canonicalizes
            real_path = Path(os.path.realpath(v))

            # 3. Containment Check
            if not real_path.is_relative_to(cwd):
                msg = f"Potential path must be strictly within current working directory: {cwd}"
                raise ValueError(msg)

        except Exception as e:
            msg = f"Invalid potential path: {v}. Error: {e}"
            raise ValueError(msg) from e

        return v
