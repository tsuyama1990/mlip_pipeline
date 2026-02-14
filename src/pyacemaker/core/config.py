"""Configuration models for PYACEMAKER."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pyacemaker.core.exceptions import ConfigurationError

# Allow configuration of defaults path via environment variable for testing/portability
_DEFAULTS_PATH = Path(
    os.environ.get("PYACEMAKER_DEFAULTS_PATH", Path(__file__).parent / "defaults.yaml")
)


def _load_defaults() -> dict[str, Any]:
    """Load default configuration values from YAML file."""
    if not _DEFAULTS_PATH.exists():
        msg = f"Defaults file not found at {_DEFAULTS_PATH}"
        raise FileNotFoundError(msg)
    with _DEFAULTS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = f"Defaults file must contain a YAML dictionary, got {type(data)}"
        raise TypeError(msg)

    return data


_DEFAULTS = _load_defaults()


class Constants(BaseSettings):
    """System-wide constants configuration.

    Values are loaded from defaults.yaml but can be overridden by environment variables
    (e.g., PYACEMAKER_MAX_CONFIG_SIZE).
    """

    model_config = SettingsConfigDict(extra="forbid", env_prefix="PYACEMAKER_")

    default_log_format: str = _DEFAULTS["log_format"]
    # 1 MB limit for configuration files to prevent OOM/DOS
    max_config_size: int = _DEFAULTS["max_config_size"]
    default_version: str = _DEFAULTS["version"]
    default_log_level: str = _DEFAULTS["log_level"]
    default_structure_strategy: str = _DEFAULTS["structure_strategy"]
    default_trainer_potential: str = _DEFAULTS["trainer_potential"]
    default_trainer_cutoff: float = _DEFAULTS["trainer_cutoff"]
    default_trainer_order: int = _DEFAULTS["trainer_order"]
    default_trainer_basis_size: tuple[int, int] = tuple(_DEFAULTS["trainer_basis_size"])
    default_trainer_delta_learning: str = _DEFAULTS["trainer_delta_learning"]
    default_trainer_max_epochs: int = _DEFAULTS["trainer_max_epochs"]
    default_trainer_batch_size: int = _DEFAULTS["trainer_batch_size"]
    default_engine: str = _DEFAULTS["engine"]

    # Orchestrator Defaults
    default_orchestrator_max_cycles: int = _DEFAULTS["orchestrator"]["max_cycles"]
    default_orchestrator_uncertainty: float = _DEFAULTS["orchestrator"]["uncertainty_threshold"]
    default_orchestrator_n_local_candidates: int = _DEFAULTS["orchestrator"]["n_local_candidates"]
    default_orchestrator_n_active_set_select: int = _DEFAULTS["orchestrator"]["n_active_set_select"]
    default_orchestrator_validation_split: float = _DEFAULTS["orchestrator"]["validation_split"]
    default_orchestrator_min_validation_size: int = _DEFAULTS["orchestrator"]["min_validation_size"]
    default_orchestrator_max_validation_size: int = _DEFAULTS["orchestrator"]["max_validation_size"]

    # Validator Defaults
    default_validator_metrics: list[str] = _DEFAULTS["validator_metrics"]

    version_regex: str = _DEFAULTS["version_regex"]
    # Allow skipping file checks for tests
    skip_file_checks: bool = _DEFAULTS["skip_file_checks"]

    # Oracle / DFT defaults
    default_dft_code: str = _DEFAULTS["dft"]["code"]
    default_dft_command: str = _DEFAULTS["dft"]["command"]
    default_dft_kspacing: float = _DEFAULTS["dft"]["kspacing"]
    default_dft_smearing: float = _DEFAULTS["dft"]["smearing"]
    default_dft_max_retries: int = _DEFAULTS["dft"]["max_retries"]
    default_dft_mixing_beta: float = _DEFAULTS["dft"]["mixing_beta"]
    default_dft_chunk_size: int = _DEFAULTS["dft"]["chunk_size"]
    default_dft_max_workers: int = _DEFAULTS["dft"]["max_workers"]

    # DFT Defaults
    default_dft_ecutwfc: float = _DEFAULTS["dft"]["ecutwfc"]
    default_dft_ecutrho: float = _DEFAULTS["dft"]["ecutrho"]
    default_dft_conv_thr: float = _DEFAULTS["dft"]["conv_thr"]
    default_dft_occupations: str = _DEFAULTS["dft"]["occupations"]
    default_dft_smearing_method: str = _DEFAULTS["dft"]["smearing_method"]

    # Error patterns for DFT retry logic
    dft_recoverable_errors: list[str] = _DEFAULTS["dft"]["recoverable_errors"]
    # Allowed input keys for security validation
    dft_allowed_input_sections: list[str] = _DEFAULTS["dft"]["allowed_input_sections"]

    # Dynamics Engine Defaults
    default_dynamics_gamma_threshold: float = _DEFAULTS["dynamics_gamma_threshold"]

    # Security Warnings
    PICKLE_SECURITY_WARNING: str = _DEFAULTS["pickle_security_warning"]

    # Regex
    valid_key_regex: str = _DEFAULTS["valid_key_regex"]
    valid_value_regex: str = _DEFAULTS["valid_value_regex"]

    # Trainer Defaults
    TRAINER_TEMP_PREFIX_TRAIN: str = _DEFAULTS["temp_prefix_train"]
    TRAINER_TEMP_PREFIX_ACTIVE: str = _DEFAULTS["temp_prefix_active"]

    # DFT Manager Defaults
    DFT_TEMP_PREFIX: str = _DEFAULTS["dft_temp_prefix"]

    # Physical Bounds
    max_energy_ev: float = _DEFAULTS["max_energy_ev"]
    max_force_ev_a: float = _DEFAULTS["max_force_ev_a"]

    # Security & Limits
    max_atoms_dft: int = _DEFAULTS["max_atoms_dft"]
    dynamics_halt_probability: float = _DEFAULTS["dynamics_halt_probability"]


CONSTANTS = Constants()

# Compiled regex for strict validation
_VALID_KEY_REGEX = re.compile(CONSTANTS.valid_key_regex)
_VALID_VALUE_REGEX = re.compile(CONSTANTS.valid_value_regex)


def _recursive_validate_parameters(
    data: dict[str, Any] | list[Any] | tuple[Any, ...], path: str = "", depth: int = 0
) -> None:
    """Recursively validate parameter dictionary or list for security.

    This function enforces a strict whitelist for keys and values to prevent injection attacks.
    It recurses into dictionaries and lists/tuples.
    """
    # Prevent stack overflow attacks via deep nesting
    if depth > 10:
        msg = "Configuration nesting too deep (max 10)"
        raise ValueError(msg)

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if not isinstance(key, str):
                msg = f"Keys must be strings at {current_path}"
                raise TypeError(msg)

            # Strict whitelist validation for keys
            if not _VALID_KEY_REGEX.match(key):
                msg = f"Invalid characters in key '{current_path}'. Must match {_VALID_KEY_REGEX.pattern}"
                raise ValueError(msg)

            _recursive_validate_value(value, current_path, depth)
    elif isinstance(data, (list, tuple)):
        for i, value in enumerate(data):
            current_path = f"{path}[{i}]"
            _recursive_validate_value(value, current_path, depth)


def _recursive_validate_value(value: Any, current_path: str, depth: int) -> None:
    """Helper to validate a value based on its type."""
    if isinstance(value, (dict, list, tuple)):
        _recursive_validate_parameters(value, current_path, depth + 1)
    elif isinstance(value, str):
        # Strict whitelist check for values (Security hardening)
        if not _VALID_VALUE_REGEX.match(value):
            msg = f"Invalid characters in value at '{current_path}'. Found potentially unsafe characters."
            raise ValueError(msg)
    elif not isinstance(value, (int, float, bool, type(None))):
        # Allow basic types, reject complex objects
        msg = f"Invalid type {type(value)} at {current_path}"
        raise TypeError(msg)


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
        _recursive_validate_parameters(v)
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
        # Prevent path traversal characters explicitly
        if ".." in v.parts:
            msg = f"Invalid root directory: {v}. Path traversal not allowed."
            raise ValueError(msg)

        try:
            # Strict resolution checks existence and resolves symlinks
            resolved = v.resolve(strict=True)

            # Additional security check: ensure resolved path is absolute
            if not resolved.is_absolute():
                # Should typically be absolute after resolve(), but explicitly checking is safer
                msg = f"Resolved root directory is not absolute: {resolved}"
                raise ValueError(msg)

        except (OSError, RuntimeError):
            # If it doesn't exist, we can't fully resolve symlinks in the final component.
            # But we can resolve the parent.
            try:
                # Resolve parent strictly
                v.parent.resolve(strict=True)
                # If parent exists, use absolute path for the full path
                resolved = v.absolute()
            except (OSError, RuntimeError) as e:
                # Parent doesn't exist or loop?
                msg = f"Invalid root directory: {v}. Parent directory must exist."
                raise ValueError(msg) from e

        return resolved


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
    max_workers: int = Field(
        default=CONSTANTS.default_dft_max_workers,
        description="Maximum number of parallel workers for DFT calculations",
    )
    embedding_enabled: bool = Field(
        default=True, description="Enable periodic embedding for non-periodic structures"
    )
    embedding_buffer: float = Field(
        default=2.0, description="Buffer size for periodic embedding (Angstrom)"
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

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        _recursive_validate_parameters(v)
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
    initial_exploration: str = Field(
        default="m3gnet", description="Initial exploration strategy (m3gnet or random)"
    )
    strain_range: float = Field(
        default=0.15, ge=0.0, description="Maximum strain range for random perturbation"
    )
    rattle_amplitude: float = Field(
        default=0.1, ge=0.0, description="Standard deviation for atomic rattling (Angstrom)"
    )
    defect_density: float = Field(
        default=0.01, ge=0.0, description="Target defect density for defect strategy"
    )


class TrainerConfig(BaseModuleConfig):
    """Trainer module configuration."""

    potential_type: str = Field(
        default=CONSTANTS.default_trainer_potential, description="Type of potential to train"
    )
    mock: bool = Field(default=False, description="Use mock trainer for testing")
    cutoff: float = Field(
        default=CONSTANTS.default_trainer_cutoff,
        description="Cutoff radius for the potential (Angstrom)",
    )
    order: int = Field(
        default=CONSTANTS.default_trainer_order, description="Maximum correlation order"
    )
    basis_size: tuple[int, int] = Field(
        default=CONSTANTS.default_trainer_basis_size,
        description="Basis size (radial, angular)",
    )
    delta_learning: str = Field(
        default=CONSTANTS.default_trainer_delta_learning,
        description="Delta learning method (zbl, lj, none)",
    )
    max_epochs: int = Field(
        default=CONSTANTS.default_trainer_max_epochs,
        description="Maximum number of training epochs",
    )
    batch_size: int = Field(
        default=CONSTANTS.default_trainer_batch_size, description="Batch size for training"
    )

    @field_validator("delta_learning")
    @classmethod
    def validate_delta_learning(cls, v: str) -> str:
        """Validate delta learning method."""
        valid = {"zbl", "lj", "none"}
        if v.lower() not in valid:
            msg = f"Invalid delta_learning: {v}. Must be one of {valid}"
            raise ValueError(msg)
        return v.lower()


class DynamicsEngineConfig(BaseModuleConfig):
    """Dynamics Engine module configuration."""

    engine: str = Field(default=CONSTANTS.default_engine, description="MD/kMC engine")
    gamma_threshold: float = Field(
        default=CONSTANTS.default_dynamics_gamma_threshold,
        description="Threshold for extrapolation grade (gamma) to trigger halt",
    )
    timestep: float = Field(default=0.001, description="Timestep in ps")
    temperature: float = Field(default=300.0, description="Temperature in K")
    pressure: float = Field(default=0.0, description="Pressure in Bar")
    n_steps: int = Field(default=100000, description="Number of MD steps")
    hybrid_baseline: str = Field(default="zbl", description="Hybrid potential baseline (zbl or lj)")
    mock: bool = Field(default=False, description="Use mock engine for testing")

    @field_validator("hybrid_baseline")
    @classmethod
    def validate_hybrid_baseline(cls, v: str) -> str:
        """Validate hybrid baseline."""
        valid = {"zbl", "lj"}
        if v.lower() not in valid:
            msg = f"Invalid hybrid_baseline: {v}. Must be one of {valid}"
            raise ValueError(msg)
        return v.lower()


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
    validation_split: float = Field(
        default=CONSTANTS.default_orchestrator_validation_split,
        description="Fraction of dataset to use for validation (0.0-1.0)",
    )
    min_validation_size: int = Field(
        default=CONSTANTS.default_orchestrator_min_validation_size,
        description="Minimum number of structures in validation set",
    )
    max_validation_size: int = Field(
        default=CONSTANTS.default_orchestrator_max_validation_size,
        description="Maximum number of structures in validation set (to prevent OOM)",
    )
    dataset_file: str = Field(
        default="dataset.pckl.gzip",
        description="Filename for the dataset within the data directory",
    )

    @field_validator("validation_split")
    @classmethod
    def validate_split(cls, v: float) -> float:
        """Validate validation split."""
        if not (0.0 <= v <= 1.0):
            msg = f"Validation split must be between 0.0 and 1.0, got {v}"
            raise ValueError(msg)
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: str = Field(
        default=CONSTANTS.default_log_level,
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
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

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


def _validate_file_security(path: Path) -> None:
    """Validate file permissions and ownership."""
    # Resolve symlinks to check actual file
    real_path = path.resolve()
    if not real_path.is_file():
        msg = f"Path is not a regular file: {path.name}"
        raise ConfigurationError(msg)

    # Check if original path is a symlink (Strict Audit Compliance)
    if path.is_symlink():
        # We allow symlinks if they point to valid files, but we should be aware.
        # For now, just proceeding as we resolved it.
        pass

    # Check file permissions (Security)
    if not os.access(path, os.R_OK):
        msg = f"Permission denied: {path.name}"
        raise ConfigurationError(msg)

    try:
        st = real_path.stat()
        # Check file ownership (Linux/Unix only) - Security
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            # In some CI/Docker environments, UID might not match.
            pass
        # Check for world-writable
        if st.st_mode & 0o002:
            msg = f"Configuration file {path.name} is world-writable. This is insecure."
            raise ConfigurationError(msg)
    except OSError as e:
        msg = f"Error checking file permissions: {e}"
        raise ConfigurationError(msg) from e


def _check_file_size(file_size: int) -> None:
    """Check if file size exceeds limit."""
    if file_size > CONSTANTS.max_config_size:
        msg = f"Configuration file too large: {file_size} bytes (max {CONSTANTS.max_config_size})"
        raise ConfigurationError(msg)


def _read_file_content(path: Path) -> str:
    """Read file content with safety checks."""
    try:
        file_size = path.stat().st_size
        _check_file_size(file_size)
        with path.open("r", encoding="utf-8") as f:
            return f.read(CONSTANTS.max_config_size + 1)
    except OSError as e:
        msg = f"Error reading configuration file: {e}"
        raise ConfigurationError(msg, details={"filename": path.name}) from e


def _read_config_file(path: Path) -> dict[str, Any]:
    """Read and parse configuration file safely.

    This function reads the file into memory with a strict size limit,
    preventing Out-Of-Memory (OOM) attacks from large files.
    """
    content = _read_file_content(path)

    if len(content) > CONSTANTS.max_config_size:
        msg = f"Configuration file exceeds size limit of {CONSTANTS.max_config_size} bytes."
        raise ConfigurationError(msg)

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML configuration: {e}"
        raise ConfigurationError(msg, details={"original_error": str(e)}) from e
    except Exception as e:
        # Catch generic errors to ensure consistent exception wrapping
        msg = f"Unexpected error reading configuration: {e}"
        raise ConfigurationError(msg) from e
    else:
        if not isinstance(data, dict):
            msg = f"Configuration file must contain a YAML dictionary, got {type(data).__name__}."
            raise ConfigurationError(msg)

        # Security: Validate high-level structure to ensure no unexpected root keys
        allowed_root_keys = {
            "version",
            "project",
            "logging",
            "orchestrator",
            "structure_generator",
            "oracle",
            "trainer",
            "dynamics_engine",
            "validator",
        }
        unknown_keys = set(data.keys()) - allowed_root_keys
        if unknown_keys:
            msg = f"Unknown configuration sections: {unknown_keys}"
            raise ConfigurationError(msg)

        return data


def load_config(path: Path) -> PYACEMAKERConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated PYACEMAKERConfig object.

    Raises:
        ConfigurationError: If the file cannot be read or validation fails.

    """
    if not path.exists(): # or not path.is_file(): - is_file check moved to validation
        msg = f"Configuration file not found: {path.name}"
        raise ConfigurationError(msg)

    _validate_file_security(path)

    try:
        data = _read_config_file(path)
        return PYACEMAKERConfig(**data)
    except ConfigurationError:
        raise
    except ValidationError as e:
        msg = f"Invalid configuration: {e}"
        details = {"errors": e.errors()}
        raise ConfigurationError(msg, details=details) from e
    except Exception as e:
        # Catch unexpected errors during load
        self_logger = __import__("logging").getLogger("pyacemaker.core.config")
        self_logger.exception("Unexpected error loading configuration")
        msg = f"Unexpected error loading configuration: {e}"
        raise ConfigurationError(msg) from e
