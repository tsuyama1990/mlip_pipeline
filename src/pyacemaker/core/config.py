"""Configuration models for PYACEMAKER."""

import re
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
    # 1 MB limit for configuration files to prevent OOM/DOS
    max_config_size: int = 1 * 1024 * 1024
    default_version: str = "0.1.0"
    default_log_level: str = "INFO"
    default_structure_strategy: str = "random"
    default_trainer_potential: str = "pace"
    default_engine: str = "lammps"
    default_orchestrator_max_cycles: int = 10
    default_orchestrator_uncertainty: float = 0.1
    default_orchestrator_n_local_candidates: int = 10
    default_orchestrator_n_active_set_select: int = 5
    default_validator_metrics: list[str] = ["rmse_energy", "rmse_forces"]
    version_regex: str = r"^\d+\.\d+\.\d+$"
    default_dft_code: str = "quantum_espresso"
    default_dft_parameters: dict[str, Any] = {}
    default_oracle_mock: bool = False


CONSTANTS = Constants()


class ProjectConfig(BaseModel):
    """Project-level configuration settings.

    Attributes:
        name: The name of the project.
        root_dir: The root directory for project outputs and artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the project")
    root_dir: Path = Field(..., description="Root directory of the project")

    @field_validator("root_dir")
    @classmethod
    def validate_root_dir(cls, v: Path) -> Path:
        """Validate root directory for path traversal and existence.

        Ensures the path does not contain '..' traversal components and attempts
        to resolve it to an absolute path.
        """
        # Check components before resolve to catch explicit ".."
        if ".." in v.parts:
            msg = f"Invalid root directory: {v}. Path traversal not allowed."
            raise ValueError(msg)

        try:
            # strictly resolve if it exists to prevent symlink attacks
            if v.exists():
                return v.resolve(strict=True)
            # if it doesn't exist, we resolve relative to cwd
            return v.resolve(strict=False)
        except OSError as e:
            # If resolution fails due to permissions or loop, reject
            msg = f"Invalid root directory path: {e}"
            raise ValueError(msg) from e


class DFTConfig(BaseModel):
    """Configuration for Density Functional Theory (DFT) calculations.

    Attributes:
        code: The DFT code to use (e.g., 'quantum_espresso', 'vasp').
        parameters: A dictionary of parameters to pass to the DFT code.
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        default=CONSTANTS.default_dft_code,
        description="DFT code to use (e.g., 'quantum_espresso', 'vasp')"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="DFT calculation parameters"
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        if not all(isinstance(k, str) for k in v):
            msg = "Parameter keys must be strings"
            raise ValueError(msg)
        return v


class OracleConfig(BaseModel):
    """Configuration for the Oracle module.

    Attributes:
        dft: DFT configuration settings.
        mock: Whether to use a mock oracle for testing/simulation.
    """

    model_config = ConfigDict(extra="forbid")

    dft: DFTConfig = Field(..., description="DFT configuration")
    mock: bool = Field(
        default=CONSTANTS.default_oracle_mock, description="Use mock oracle for testing"
    )


class StructureGeneratorConfig(BaseModel):
    """Configuration for the Structure Generator module.

    Attributes:
        strategy: The strategy name for structure generation.
        parameters: Strategy-specific parameters.
    """

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
        """Validate parameters dictionary."""
        if not all(isinstance(k, str) for k in v):
            msg = "Parameter keys must be strings"
            raise ValueError(msg)
        return v


class TrainerConfig(BaseModel):
    """Configuration for the Trainer module.

    Attributes:
        potential_type: The type of potential to train (e.g., 'pace').
        parameters: Training parameters.
    """

    model_config = ConfigDict(extra="forbid")

    potential_type: str = Field(
        default=CONSTANTS.default_trainer_potential,
        description="Type of potential to train"
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Training parameters")

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        if not all(isinstance(k, str) for k in v):
            msg = "Parameter keys must be strings"
            raise ValueError(msg)
        return v


class DynamicsEngineConfig(BaseModel):
    """Configuration for the Dynamics Engine module.

    Attributes:
        engine: The engine name (e.g., 'lammps').
        parameters: Engine-specific parameters.
    """

    model_config = ConfigDict(extra="forbid")

    engine: str = Field(
        default=CONSTANTS.default_engine,
        description="MD/kMC engine"
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Engine parameters")

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        if not all(isinstance(k, str) for k in v):
            msg = "Parameter keys must be strings"
            raise ValueError(msg)
        return v


class ValidatorConfig(BaseModel):
    """Configuration for the Validator module.

    Attributes:
        metrics: List of metrics to evaluate.
        thresholds: Threshold values for metrics.
    """

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default_factory=lambda: CONSTANTS.default_validator_metrics,
        description="Metrics to validate",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict, description="Validation thresholds"
    )


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator.

    Attributes:
        max_cycles: Maximum number of active learning cycles.
        uncertainty_threshold: Threshold for triggering learning.
        n_local_candidates: Number of candidates to generate per seed.
        n_active_set_select: Number of structures to select for the active set.
    """

    model_config = ConfigDict(extra="forbid")

    max_cycles: int = Field(
        default=CONSTANTS.default_orchestrator_max_cycles,
        description="Maximum number of active learning cycles"
    )
    uncertainty_threshold: float = Field(
        default=CONSTANTS.default_orchestrator_uncertainty,
        description="Threshold for uncertainty sampling"
    )
    n_local_candidates: int = Field(
        default=CONSTANTS.default_orchestrator_n_local_candidates,
        description="Number of local candidates to generate per seed"
    )
    n_active_set_select: int = Field(
        default=CONSTANTS.default_orchestrator_n_active_set_select,
        description="Number of structures to select for active set"
    )


class LoggingConfig(BaseModel):
    """Logging configuration settings.

    Attributes:
        level: Logging level (DEBUG, INFO, etc.).
        format: Format string for log messages.
    """

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
    """Root configuration object for the PYACEMAKER application.

    Attributes:
        version: Configuration schema version.
        project: Project-specific settings.
        logging: Logging settings.
        orchestrator: Orchestrator settings.
        structure_generator: Structure generator settings.
        oracle: Oracle settings.
        trainer: Trainer settings.
        dynamics_engine: Dynamics engine settings.
        validator: Validator settings.
    """

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

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        if not re.match(CONSTANTS.version_regex, v):
            msg = f"Invalid version format: {v}. Must match {CONSTANTS.version_regex}"
            raise ValueError(msg)
        return v


def load_config(path: Path) -> PYACEMAKERConfig:
    """Load and validate configuration from a YAML file.

    This function enforces strict size limits on the input file to prevent
    Out-Of-Memory (OOM) attacks or accidents. It reads the file into memory
    only if it is below the defined threshold.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated PYACEMAKERConfig object.

    Raises:
        ConfigurationError: If the file cannot be read, is too large, or validation fails.

    """
    if not path.exists():
        msg = f"Configuration file not found: {path.name}"
        raise ConfigurationError(msg)

    if not path.is_file():
        msg = f"Configuration path is not a file: {path.name}"
        raise ConfigurationError(msg)

    # Check file size to prevent DOS (OOM) - Preliminary check
    try:
        if path.stat().st_size > CONSTANTS.max_config_size:
            msg = f"Configuration file too large: {path.stat().st_size} bytes (max {CONSTANTS.max_config_size} bytes)"
            raise ConfigurationError(msg)
    except OSError as e:
        msg = f"Error checking configuration file size: {e}"
        raise ConfigurationError(msg, details={"filename": path.name}) from e

    try:
        with path.open("r", encoding="utf-8") as f:
            # Safe loading with hard limit on read size to handle race conditions
            content = f.read(CONSTANTS.max_config_size + 1)

            if len(content) > CONSTANTS.max_config_size:
                msg = f"Configuration file content exceeds limit of {CONSTANTS.max_config_size} bytes"
                raise ConfigurationError(msg)

            data = yaml.safe_load(content)

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
