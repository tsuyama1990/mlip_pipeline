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
    def validate_root_dir(cls, v: Path) -> Path:
        """Validate root directory for path traversal."""
        # Check components before resolve to catch explicit ".."
        if ".." in v.parts:
            msg = f"Invalid root directory: {v}. Path traversal not allowed."
            raise ValueError(msg)
        return v.resolve()


class DFTConfig(BaseModel):
    """DFT calculation configuration."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="DFT code to use (e.g., 'quantum_espresso', 'vasp')")


class Constants(BaseModel):
    """System-wide constants configuration."""

    default_log_format: str = "[{time}] [{level}] [{extra[name]}] {message}"
    max_config_size: int = 10 * 1024 * 1024  # 10 MB


CONSTANTS = Constants()


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    format: str = Field(default=CONSTANTS.default_log_format, description="Log message format")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            msg = f"Invalid log level: {v}. Must be one of {valid}"
            raise ValueError(msg)
        return v.upper()


class PYACEMAKERConfig(BaseModel):
    """Main configuration for the application."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(
        "0.1.0", description="Configuration schema version", pattern=r"^\d+\.\d+\.\d+$"
    )
    project: ProjectConfig
    dft: DFTConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)  # type: ignore[arg-type]


def load_config(path: Path) -> PYACEMAKERConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated PYACEMAKERConfig object.

    Raises:
        ConfigurationError: If the file cannot be read or validation fails.

    """
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise ConfigurationError(msg)

    if path.stat().st_size > CONSTANTS.max_config_size:
        msg = f"Configuration file too large: {path.stat().st_size} bytes (max {CONSTANTS.max_config_size} bytes)"
        raise ConfigurationError(msg)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML configuration: {e}"
        raise ConfigurationError(msg, details={"original_error": str(e)}) from e
    except OSError as e:
        msg = f"Error reading configuration file: {e}"
        raise ConfigurationError(msg, details={"filename": str(path)}) from e

    if not isinstance(data, dict):
        msg = "Configuration file must contain a YAML dictionary."
        raise ConfigurationError(msg, details={"actual_type": type(data).__name__})

    try:
        return PYACEMAKERConfig(**data)
    except ValidationError as e:
        msg = f"Invalid configuration: {e}"
        details = {"errors": e.errors()}
        raise ConfigurationError(msg, details=details) from e
