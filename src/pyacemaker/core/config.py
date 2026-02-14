"""Configuration models for PYACEMAKER."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pyacemaker.core.exceptions import ConfigurationError


class Constants(BaseSettings):
    """System-wide constants configuration.

    Values can be overridden by environment variables (e.g., PYACEMAKER_MAX_CONFIG_SIZE).
    """

    model_config = SettingsConfigDict(extra="forbid", env_prefix="PYACEMAKER_")

    default_log_format: str = "[{time}] [{level}] [{extra[name]}] {message}"
    max_config_size: int = 1 * 1024 * 1024  # 1 MB limit for safety
    default_version: str = "0.1.0"
    default_log_level: str = "INFO"
    default_structure_strategy: str = "random"
    default_trainer_potential: str = "pace"
    default_engine: str = "lammps"
    default_orchestrator_max_cycles: int = 10
    default_orchestrator_uncertainty: float = 0.1
    default_validator_metrics: list[str] = ["rmse_energy", "rmse_forces"]
    version_regex: str = r"^\d+\.\d+\.\d+$"


CONSTANTS = Constants()


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

        try:
            # Return absolute path
            return v.resolve()
        except OSError:
            # If resolution fails (rare), return original but we've checked for ..
            return v


class DFTConfig(BaseModel):
    """DFT calculation configuration."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="DFT code to use (e.g., 'quantum_espresso', 'vasp')")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="DFT calculation parameters"
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        # Pydantic handles basic type checking (dict[str, Any]), but we can add extra logic if needed.
        # If Pydantic already checks keys are strings, this validator runs after that check passes
        # or if Pydantic converts them.
        return v


class OracleConfig(BaseModel):
    """Oracle module configuration."""

    model_config = ConfigDict(extra="forbid")

    dft: DFTConfig = Field(..., description="DFT configuration")
    mock: bool = Field(default=False, description="Use mock oracle for testing")


class StructureGeneratorConfig(BaseModel):
    """Structure Generator module configuration."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(
        default=CONSTANTS.default_structure_strategy,
        description="Generation strategy (e.g., 'random', 'adaptive')"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        return v


class TrainerConfig(BaseModel):
    """Trainer module configuration."""

    model_config = ConfigDict(extra="forbid")

    potential_type: str = Field(
        default=CONSTANTS.default_trainer_potential,
        description="Type of potential to train"
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Training parameters")

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        return v


class DynamicsEngineConfig(BaseModel):
    """Dynamics Engine module configuration."""

    model_config = ConfigDict(extra="forbid")

    engine: str = Field(
        default=CONSTANTS.default_engine,
        description="MD/kMC engine"
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Engine parameters")

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        return v


class ValidatorConfig(BaseModel):
    """Validator module configuration."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default_factory=lambda: CONSTANTS.default_validator_metrics,
        description="Metrics to validate",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict, description="Validation thresholds"
    )


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration."""

    model_config = ConfigDict(extra="forbid")

    max_cycles: int = Field(
        default=CONSTANTS.default_orchestrator_max_cycles,
        description="Maximum number of active learning cycles"
    )
    uncertainty_threshold: float = Field(
        default=CONSTANTS.default_orchestrator_uncertainty,
        description="Threshold for uncertainty sampling"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: str = Field(
        CONSTANTS.default_log_level,
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
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
        CONSTANTS.default_version,
        description="Configuration schema version",
        pattern=CONSTANTS.version_regex,
    )
    project: ProjectConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)  # type: ignore[arg-type]

    # Module configurations
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    structure_generator: StructureGeneratorConfig = Field(default_factory=StructureGeneratorConfig)
    oracle: OracleConfig = Field(..., description="Oracle configuration")
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    dynamics_engine: DynamicsEngineConfig = Field(default_factory=DynamicsEngineConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)


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
        msg = f"Configuration file not found: {path.name}"
        raise ConfigurationError(msg)

    if not path.is_file():
        msg = f"Configuration path is not a file: {path.name}"
        raise ConfigurationError(msg)

    # Check file size to prevent DOS (OOM)
    try:
        if path.stat().st_size > CONSTANTS.max_config_size:
            msg = f"Configuration file too large: {path.stat().st_size} bytes (max {CONSTANTS.max_config_size} bytes)"
            raise ConfigurationError(msg)
    except OSError as e:
        msg = f"Error checking configuration file size: {e}"
        raise ConfigurationError(msg, details={"filename": path.name}) from e

    # yaml.safe_load is used to prevent arbitrary code execution.
    # The file size check above mitigates OOM risks (DoS).
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML configuration: {e}"
        raise ConfigurationError(msg, details={"original_error": str(e)}) from e
    except OSError as e:
        msg = f"Error reading configuration file: {e}"
        raise ConfigurationError(msg, details={"filename": path.name}) from e

    if not isinstance(data, dict):
        msg = "Configuration file must contain a YAML dictionary."
        raise ConfigurationError(msg, details={"actual_type": type(data).__name__})

    try:
        return PYACEMAKERConfig(**data)
    except ValidationError as e:
        msg = f"Invalid configuration: {e}"
        details = {"errors": e.errors()}
        raise ConfigurationError(msg, details=details) from e
