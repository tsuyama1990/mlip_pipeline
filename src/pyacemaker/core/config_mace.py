"""MACE model configuration."""

from pydantic import BaseModel, ConfigDict, Field, field_validator
import re
from pathlib import Path
from typing import Any
from pyacemaker.core.validation import validate_safe_path, _validate_absolute_path
from pyacemaker.core.config import CONSTANTS, get_defaults

class MaceConfig(BaseModel):
    """MACE model configuration."""

    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(
        default=CONSTANTS.default_mace_model_path,
        description="Path or URL to the MACE model (e.g., 'medium', path/to/model.model)",
    )
    device: str = Field(
        default=CONSTANTS.default_mace_device, description="Device to run on (cpu, cuda)"
    )
    default_dtype: str = Field(
        default=CONSTANTS.default_mace_dtype, description="Default data type (float32, float64)"
    )
    batch_size: int = Field(
        default=CONSTANTS.default_mace_batch_size, description="Batch size for prediction"
    )
    mock: bool = Field(
        default_factory=lambda: get_defaults()["mace_mock"],
        description="Mock MACE for testing"
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str, _info: Any) -> str:
        """Validate model path existence or URL format."""
        if v.lower() in {"medium", "small", "large"}:
            return v

        # Check if URL
        if v.startswith(("http://", "https://")):
            # Strict URL validation regex to prevent injection
            if not re.match(r'^(http|https)://[a-zA-Z0-9\-\.]+(:\d+)?(/[\w\-\./:\+,=@]+)?$', v):
                 msg = f"Invalid model URL format: {v}"
                 raise ValueError(msg)
            return v

        path = Path(v)
        # Security check: ALWAYS valid path structure and traversal prevention
        try:
            # We must validate the resolved path if relative, or just validate safety
            # validate_safe_path now handles resolution and whitelist checking internally
            # but we should ensure we return the safe, resolved path if it was relative
            # to avoid usage of unsafe relative path later.

            # Note: validate_safe_path returns a path object.
            # If v is relative, validate_safe_path resolves it against CWD.
            # We should probably return the resolved absolute path string to be safe.

            safe_path = validate_safe_path(path)

            # The feedback said: "Always resolve relative paths to absolute and validate against whitelist before use."
            # validate_safe_path does exactly this:
            # 1. Resolves against CWD if relative.
            # 2. Checks against whitelist/CWD.
            # So calling it is correct. We just need to return the absolute string.

            return str(safe_path.resolve())

        except (ValueError, RuntimeError) as e:
            msg = f"Invalid model path structure: {e}"
            raise ValueError(msg) from e

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device."""
        if v not in {"cpu", "cuda", "mps"}:
            msg = f"Invalid device: {v}. Must be cpu, cuda, or mps"
            raise ValueError(msg)
        return v

    @field_validator("default_dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Validate data type."""
        if v not in {"float32", "float64"}:
            msg = f"Invalid dtype: {v}. Must be float32 or float64"
            raise ValueError(msg)
        return v
