"""Configuration models for PYACEMAKER."""

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pyacemaker.core.exceptions import ConfigurationError
from pyacemaker.core.utils import LimitedStream


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
    # Allow skipping file checks for tests
    skip_file_checks: bool = False

    # Oracle / DFT defaults
    default_dft_code: str = "quantum_espresso"
    default_dft_command: str = "mpirun -np 4 pw.x"
    default_dft_kspacing: float = 0.04
    default_dft_smearing: float = 0.02
    default_dft_max_retries: int = 3
    default_dft_mixing_beta: float = 0.7
    default_dft_chunk_size: int = 100
    # Error patterns for DFT retry logic
    dft_recoverable_errors: list[str] = [
        "scf not converged",
        "convergence not achieved",
        "electronic convergence failed",
    ]
    # Allowed input keys for security validation
    dft_allowed_input_sections: list[str] = ["control", "system", "electrons", "ions", "cell"]
    # Security Warnings
    PICKLE_SECURITY_WARNING: str = (
        "SECURITY WARNING: pickle is unsafe. Do not load untrusted data. "
        "Ensure datasets are from trusted sources."
    )


CONSTANTS = Constants()


class BaseModuleConfig(BaseModel):
    """Base configuration for modules with parameters."""

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Module-specific parameters"
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        # Ensure keys are strings
        if not all(isinstance(k, str) for k in v):
            msg = "Parameter keys must be strings"
            raise ValueError(msg)
        return v


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the project")
    root_dir: Path = Field(..., description="Root directory of the project")

    @field_validator("root_dir")
    @classmethod
    def validate_root_dir(cls, v: Path) -> Path:
        """Validate root directory for path traversal."""
        try:
            # Strict resolution checks existence and resolves symlinks
            return v.resolve(strict=True)
        except OSError:
            # If path doesn't exist yet (e.g., initial setup), check for traversal in parts
            if ".." in v.parts:
                msg = f"Invalid root directory: {v}. Path traversal not allowed."
                raise ValueError(msg) from None
            return v.absolute()


class DFTConfig(BaseModel):
    """DFT calculation configuration."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        default=CONSTANTS.default_dft_code, description="DFT code to use (e.g., 'quantum_espresso')"
    )
    command: str = Field(
        default=CONSTANTS.default_dft_command, description="Command to run the DFT code"
    )
    pseudo_dir: Path = Field(
        default=Path(), description="Directory containing pseudopotential files"
    )
    pseudopotentials: dict[str, str] = Field(
        ..., description="Map of element symbol to pseudopotential filename"
    )
    kspacing: float = Field(
        default=CONSTANTS.default_dft_kspacing, description="K-point spacing in inverse Angstroms"
    )
    smearing: float = Field(
        default=CONSTANTS.default_dft_smearing, description="Smearing width in eV"
    )
    max_retries: int = Field(
        default=CONSTANTS.default_dft_max_retries,
        description="Maximum number of retries for failed calculations",
    )
    chunk_size: int = Field(
        default=CONSTANTS.default_dft_chunk_size,
        description="Number of structures to process in a single batch (DFT)",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters (e.g. for mocking)"
    )

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotentials(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate existence of pseudopotential files."""
        if CONSTANTS.skip_file_checks:
            return v

        missing = []
        for element, path_str in v.items():
            path = Path(path_str)
            if not path.exists():
                missing.append(f"{element}: {path_str}")

        if missing:
            msg = f"Missing pseudopotential files: {', '.join(missing)}"
            raise ValueError(msg)
        return v


class OracleConfig(BaseModel):
    """Oracle module configuration."""

    model_config = ConfigDict(extra="forbid")

    dft: DFTConfig = Field(..., description="DFT configuration")
    mock: bool = Field(default=False, description="Use mock oracle for testing")


class StructureGeneratorConfig(BaseModuleConfig):
    """Structure Generator module configuration."""

    strategy: str = Field(
        default=CONSTANTS.default_structure_strategy,
        description="Generation strategy (e.g., 'random', 'adaptive')",
    )


class TrainerConfig(BaseModuleConfig):
    """Trainer module configuration."""

    potential_type: str = Field(
        default=CONSTANTS.default_trainer_potential, description="Type of potential to train"
    )


class DynamicsEngineConfig(BaseModuleConfig):
    """Dynamics Engine module configuration."""

    engine: str = Field(default=CONSTANTS.default_engine, description="MD/kMC engine")


class ValidatorConfig(BaseModel):
    """Validator module configuration."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default_factory=lambda: CONSTANTS.default_validator_metrics,
        description="Metrics to validate",
    )
    thresholds: dict[str, float] = Field(default_factory=dict, description="Validation thresholds")


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration."""

    model_config = ConfigDict(extra="forbid")

    max_cycles: int = Field(
        default=CONSTANTS.default_orchestrator_max_cycles,
        description="Maximum number of active learning cycles",
    )
    uncertainty_threshold: float = Field(
        default=CONSTANTS.default_orchestrator_uncertainty,
        description="Threshold for uncertainty sampling",
    )
    n_local_candidates: int = Field(
        default=CONSTANTS.default_orchestrator_n_local_candidates,
        description="Number of local candidates to generate per seed",
    )
    n_active_set_select: int = Field(
        default=CONSTANTS.default_orchestrator_n_active_set_select,
        description="Number of structures to select for active set",
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

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version."""
        if not re.match(CONSTANTS.version_regex, v):
            msg = f"Invalid version format: {v}. Must match {CONSTANTS.version_regex}"
            raise ValueError(msg)
        return v


def load_config(path: Path) -> PYACEMAKERConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated PYACEMAKERConfig object.

    Raises:
        ConfigurationError: If the file cannot be read or validation fails.

    """
    if not path.exists() or not path.is_file():
        msg = f"Configuration file not found or invalid: {path.name}"
        raise ConfigurationError(msg)

    # Check file permissions (Security)
    # Ensure file is readable by current user
    if not os.access(path, os.R_OK):
        msg = f"Permission denied: {path.name}"
        raise ConfigurationError(msg)

    try:
        # Check size hint first, though LimitedStream is the real guard
        if path.stat().st_size > CONSTANTS.max_config_size:
            msg = f"Configuration file too large: {path.stat().st_size} bytes"
            raise ConfigurationError(msg)  # noqa: TRY301

        with path.open("r", encoding="utf-8") as f:
            # Use LimitedStream to enforce size limit during parsing
            stream = LimitedStream(f, CONSTANTS.max_config_size)
            data = yaml.safe_load(stream)

    except yaml.YAMLError as e:
        msg = f"Error parsing YAML configuration: {e}"
        raise ConfigurationError(msg, details={"original_error": str(e)}) from e
    except OSError as e:
        msg = f"Error reading configuration file: {e}"
        raise ConfigurationError(msg, details={"filename": path.name}) from e
    except ConfigurationError:
        raise
    except Exception as e:  # Catch unexpected errors during load
        msg = f"Unexpected error loading configuration: {e}"
        raise ConfigurationError(msg) from e

    if not isinstance(data, dict):
        msg = "Configuration file must contain a YAML dictionary."
        raise ConfigurationError(msg, details={"actual_type": type(data).__name__})

    try:
        return PYACEMAKERConfig(**data)
    except ValidationError as e:
        msg = f"Invalid configuration: {e}"
        details = {"errors": e.errors()}
        raise ConfigurationError(msg, details=details) from e

import os
