"""Configuration models for PYACEMAKER."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    # Dataset limits
    max_object_size: int = _DEFAULTS["max_object_size"]
    default_buffer_size: int = _DEFAULTS["default_buffer_size"]

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
    default_validation_buffer_size: int = 100

    # Validator Defaults
    default_validator_metrics: list[str] = _DEFAULTS["validator_metrics"]

    version_regex: str = _DEFAULTS["version_regex"]
    # Allow skipping file checks for tests
    skip_file_checks: bool = False  # Secure default

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
    # Allowed input sections for security validation
    dft_allowed_input_sections: list[str] = _DEFAULTS["dft"]["allowed_input_sections"]

    # Structure Feature Whitelist
    # Allow common keys for atoms, forces, etc. plus 'atoms' object itself.
    allowed_feature_keys: list[str] = [
        "atoms", "forces", "stress", "energy", "virial", "dipole",
        "magmom", "charges", "momenta", "masses", "numbers", "positions",
        "cell", "pbc", "initial_magmoms", "initial_charges", "uncertainty",
        "original_id", "source"
    ]

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
    composition_tolerance: float = _DEFAULTS["composition_tolerance"]

    # Physics Validation Defaults
    physics_phonon_supercell: list[int] = _DEFAULTS["physics_phonon_supercell"]
    physics_phonon_tolerance: float = _DEFAULTS["physics_phonon_tolerance"]
    physics_eos_strain: float = _DEFAULTS["physics_eos_strain"]
    physics_eos_points: int = _DEFAULTS["physics_eos_points"]
    physics_elastic_strain: float = _DEFAULTS["physics_elastic_strain"]

    # Security & Limits
    max_atoms_dft: int = _DEFAULTS["max_atoms_dft"]
    dynamics_halt_probability: float = _DEFAULTS["dynamics_halt_probability"]

    # File Names
    default_dataset_file: str = "dataset.pckl.gzip"
    default_validation_file: str = "validation_set.pckl.gzip"
    default_training_file: str = "training_set.pckl.gzip"

    @field_validator("max_config_size")
    @classmethod
    def validate_max_config_size(cls, v: int) -> int:
        if v < 1024: # Minimum 1KB
            msg = "max_config_size must be at least 1KB"
            raise ValueError(msg)
        return v

    @field_validator("dynamics_halt_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            msg = "Probability must be between 0.0 and 1.0"
            raise ValueError(msg)
        return v

    @field_validator("valid_key_regex", "valid_value_regex")
    @classmethod
    def validate_regex(cls, v: str) -> str:
        if not v:
            msg = "Regex pattern cannot be empty"
            raise ValueError(msg)
        try:
            re.compile(v)
        except re.error as e:
            msg = f"Invalid regex pattern: {e}"
            raise ValueError(msg) from e
        return v


CONSTANTS = Constants()

# Compiled regex for strict validation
_VALID_KEY_REGEX = re.compile(CONSTANTS.valid_key_regex)
_VALID_VALUE_REGEX = re.compile(CONSTANTS.valid_value_regex)


def _check_path_containment(path: Path) -> None:
    """Check that path is within the current working directory."""
    # Explicitly disallow '..' in path parts to prevent traversal attempts
    if ".." in path.parts:
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Bypass strict containment check for tests if configured
    if CONSTANTS.skip_file_checks:
        return

    try:
        cwd = Path.cwd().resolve()
        resolved = path.resolve()

        # Security: Ensure resolved path is absolute and within CWD
        if not resolved.is_absolute():
            msg = f"Resolved path is not absolute: {resolved}"
            raise ValueError(msg)

        if not resolved.is_relative_to(cwd):
            msg = f"Path must be within current working directory: {cwd}"
            raise ValueError(msg)  # noqa: TRY301

    except (ValueError, RuntimeError) as e:
        msg = f"Path {path} is unsafe or outside allowed base directory"
        raise ValueError(msg) from e


def _validate_structure(data: Any, path: str = "", depth: int = 0) -> None:
    """Validate data structure recursively against security rules.

    Consolidated validation logic for dictionaries and lists to prevent injection attacks
    and stack overflows.
    """
    if depth > 10:
        msg = "Configuration nesting too deep (max 10)"
        raise ValueError(msg)

    if isinstance(data, dict):
        _validate_dict(data, path, depth)
    elif isinstance(data, (list, tuple)):
        _validate_list(data, path, depth)
    elif isinstance(data, str):
        if not _VALID_VALUE_REGEX.match(data):
            msg = f"Invalid characters in value at '{path}'. Found potentially unsafe characters."
            raise ValueError(msg)
    elif not isinstance(data, (int, float, bool, type(None))):
        msg = f"Invalid type {type(data)} at {path}"
        raise TypeError(msg)


def _validate_dict(data: dict[str, Any], path: str, depth: int) -> None:
    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key
        if not isinstance(key, str):
            msg = f"Keys must be strings at {current_path}"
            raise TypeError(msg)

        if not _VALID_KEY_REGEX.match(key):
            msg = f"Invalid characters in key '{current_path}'. Must match {_VALID_KEY_REGEX.pattern}"
            raise ValueError(msg)

        _validate_structure(value, current_path, depth + 1)


def _validate_list(data: list[Any] | tuple[Any, ...], path: str, depth: int) -> None:
    for i, value in enumerate(data):
        current_path = f"{path}[{i}]"
        _validate_structure(value, current_path, depth + 1)


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
        _validate_structure(v)
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
        missing = []
        for element, path_str in v.items():
            path = Path(path_str)

            # Security check for traversal - ALWAYS RUN
            try:
                _check_path_containment(path)
            except ValueError as e:
                # If skipping checks, we might ignore non-existence, but traversal is suspicious.
                # However, audit says "Always validate path containment".
                # But _check_path_containment checks if relative to CWD.
                # In tests, dummy paths might not be in CWD?
                # If skip_file_checks is True, we assume mocking.
                # But path traversal (..) should still be forbidden.
                # _check_path_containment enforces `..` check first.
                msg = f"Invalid path for {element}: {e}"
                raise ValueError(msg) from e

            if not CONSTANTS.skip_file_checks and not path.exists():
                missing.append(f"{element}: {path_str}")

        if missing:
            msg = f"Missing pseudopotential files: {', '.join(missing)}"
            raise ValueError(msg)
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        _validate_structure(v)
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


class EONConfig(BaseModel):
    """Configuration for EON kMC simulations."""

    model_config = ConfigDict(extra="forbid")

    executable: str = Field(default="eonclient", description="Path to EON executable")
    parameters: dict[str, Any] = Field(default_factory=dict, description="EON parameters")

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        _validate_structure(v)
        return v


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
    eon: EONConfig = Field(default_factory=EONConfig, description="EON configuration")

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
    test_set_ratio: float = Field(default=0.1, description="Ratio of dataset to use for testing")
    phonon_supercell: list[int] = Field(
        default_factory=lambda: CONSTANTS.physics_phonon_supercell,
        description="Supercell for phonon calculation",
    )
    eos_strain: float = Field(
        default=CONSTANTS.physics_eos_strain, description="Strain range for EOS calculation"
    )
    elastic_strain: float = Field(
        default=CONSTANTS.physics_elastic_strain,
        description="Strain for elastic constants calculation",
    )


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
        default=CONSTANTS.default_dataset_file,
        description="Filename for the dataset within the data directory",
    )
    validation_buffer_size: int = Field(
        default=CONSTANTS.default_validation_buffer_size,
        description="Buffer size for writing validation items to disk",
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
    """Main configuration for the application.

    This configuration object is the root of the hierarchical configuration system.
    It contains sub-configurations for each module:
    - project: Project-level settings (root dir, name)
    - logging: Logging settings
    - orchestrator: Active learning cycle parameters
    - structure_generator: Structure generation strategy and parameters
    - oracle: DFT calculation parameters
    - trainer: Potential training parameters
    - dynamics_engine: MD/kMC engine parameters
    - validator: Validation thresholds and metrics
    """

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


# Configuration loading logic moved to pyacemaker.core.config_loader
