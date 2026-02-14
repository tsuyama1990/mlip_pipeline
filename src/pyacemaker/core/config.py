"""Configuration models for PYACEMAKER."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pyacemaker.core.exceptions import ConfigurationError


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the project")
    root_dir: Path = Field(..., description="Root directory of the project")

    @field_validator("root_dir")
    @classmethod
    def make_absolute(cls, v: Path) -> Path:
        """Ensure path is absolute."""
        return v.resolve()


class DFTConfig(BaseModel):
    """DFT calculation configuration."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="DFT code to use (e.g., 'quantum_espresso', 'vasp')")


class PYACEMAKERConfig(BaseModel):
    """Main configuration for the application."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(
        "0.1.0", description="Configuration schema version", pattern=r"^\d+\.\d+\.\d+$"
    )
    project: ProjectConfig
    dft: DFTConfig


def load_config(path: str | Path) -> PYACEMAKERConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated PYACEMAKERConfig object.

    Raises:
        ConfigurationError: If the file cannot be read or validation fails.

    """
    path = Path(path)
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise ConfigurationError(msg)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML configuration: {e}"
        raise ConfigurationError(msg) from e
    except OSError as e:
        msg = f"Error reading configuration file: {e}"
        raise ConfigurationError(msg) from e

    if not isinstance(data, dict):
        msg = "Configuration file must contain a YAML dictionary."
        raise ConfigurationError(msg)

    try:
        return PYACEMAKERConfig(**data)
    except ValidationError as e:
        msg = f"Invalid configuration: {e}"
        raise ConfigurationError(msg) from e
