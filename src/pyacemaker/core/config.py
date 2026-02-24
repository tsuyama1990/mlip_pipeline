"""Configuration models for PYACEMAKER."""

import functools
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pyacemaker.core.validation import validate_parameters, validate_safe_path

# Allow configuration of defaults path via environment variable for testing/portability
_DEFAULTS_PATH = Path(
    os.environ.get("PYACEMAKER_DEFAULTS_PATH", Path(__file__).parent / "defaults.yaml")
)

# Safety limit for loading defaults file
MAX_DEFAULTS_SIZE = 1024 * 1024  # 1MB limit


@functools.lru_cache(maxsize=1)
def get_defaults() -> dict[str, Any]:
    """Load default configuration values from YAML file (Cached).

    Includes validation for file size to prevent OOM attacks.
    """
    if not _DEFAULTS_PATH.exists():
        msg = f"Defaults file not found at {_DEFAULTS_PATH}"
        raise FileNotFoundError(msg)

    # Size check before reading
    if _DEFAULTS_PATH.stat().st_size > MAX_DEFAULTS_SIZE:
        msg = f"Defaults file size ({_DEFAULTS_PATH.stat().st_size}) exceeds limit of {MAX_DEFAULTS_SIZE} bytes"
        raise ValueError(msg)

    with _DEFAULTS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        msg = f"Defaults file must contain a YAML dictionary, got {type(data)}"
        raise TypeError(msg)

    return data


class Constants(BaseSettings):
    """System-wide constants configuration.

    Values are loaded lazily from defaults.yaml using default_factory.
    """

    model_config = SettingsConfigDict(extra="forbid", env_prefix="PYACEMAKER_")

    default_log_format: str = Field(
        default_factory=lambda: get_defaults()["log_format"]
    )
    max_config_size: int = Field(
        default_factory=lambda: get_defaults()["max_config_size"]
    )
    max_object_size: int = Field(
        default_factory=lambda: get_defaults()["max_object_size"]
    )
    default_buffer_size: int = Field(
        default_factory=lambda: get_defaults()["default_buffer_size"]
    )

    default_version: str = Field(
        default_factory=lambda: get_defaults()["version"]
    )
    default_log_level: str = Field(
        default_factory=lambda: get_defaults()["log_level"]
    )
    default_structure_strategy: str = Field(
        default_factory=lambda: get_defaults()["structure_strategy"]
    )
    default_trainer_potential: str = Field(
        default_factory=lambda: get_defaults()["trainer_potential"]
    )
    default_trainer_cutoff: float = Field(
        default_factory=lambda: get_defaults()["trainer_cutoff"]
    )
    default_trainer_order: int = Field(
        default_factory=lambda: get_defaults()["trainer_order"]
    )
    default_trainer_basis_size: tuple[int, int] = Field(
        default_factory=lambda: tuple(get_defaults()["trainer_basis_size"])
    )
    default_trainer_delta_learning: str = Field(
        default_factory=lambda: get_defaults()["trainer_delta_learning"]
    )
    default_trainer_max_epochs: int = Field(
        default_factory=lambda: get_defaults()["trainer_max_epochs"]
    )
    default_trainer_batch_size: int = Field(
        default_factory=lambda: get_defaults()["trainer_batch_size"]
    )
    default_trainer_elements: list[str] = Field(
        default_factory=lambda: get_defaults()["trainer_default_elements"]
    )
    default_engine: str = Field(
        default_factory=lambda: get_defaults()["engine"]
    )

    # Orchestrator Defaults
    default_orchestrator_max_cycles: int = Field(
        default_factory=lambda: get_defaults()["orchestrator"]["max_cycles"]
    )
    default_orchestrator_uncertainty: float = Field(
        default_factory=lambda: get_defaults()["orchestrator"]["uncertainty_threshold"]
    )
    default_orchestrator_n_local_candidates: int = Field(
        default_factory=lambda: get_defaults()["orchestrator"]["n_local_candidates"]
    )
    default_orchestrator_n_active_set_select: int = Field(
        default_factory=lambda: get_defaults()["orchestrator"]["n_active_set_select"]
    )
    default_orchestrator_validation_split: float = Field(
        default_factory=lambda: get_defaults()["orchestrator"]["validation_split"]
    )
    default_orchestrator_min_validation_size: int = Field(
        default_factory=lambda: get_defaults()["orchestrator"]["min_validation_size"]
    )
    default_orchestrator_max_validation_size: int = Field(
        default_factory=lambda: get_defaults()["orchestrator"]["max_validation_size"]
    )
    default_validation_buffer_size: int = Field(
        default_factory=lambda: get_defaults()["default_validation_buffer_size"]
    )

    # Validator Defaults
    default_validator_metrics: list[str] = Field(
        default_factory=lambda: get_defaults()["validator_metrics"]
    )

    version_regex: str = Field(
        default_factory=lambda: get_defaults()["version_regex"]
    )

    # Updated to load from defaults
    skip_file_checks: bool = Field(
        default_factory=lambda: get_defaults()["skip_file_checks"]
    )

    # Oracle / DFT defaults
    default_dft_code: str = Field(
        default_factory=lambda: get_defaults()["dft"]["code"]
    )
    default_dft_command: str = Field(
        default_factory=lambda: get_defaults()["dft"]["command"]
    )
    default_dft_kspacing: float = Field(
        default_factory=lambda: get_defaults()["dft"]["kspacing"]
    )
    default_dft_smearing: float = Field(
        default_factory=lambda: get_defaults()["dft"]["smearing"]
    )
    default_dft_max_retries: int = Field(
        default_factory=lambda: get_defaults()["dft"]["max_retries"]
    )
    default_dft_mixing_beta: float = Field(
        default_factory=lambda: get_defaults()["dft"]["mixing_beta"]
    )
    default_dft_chunk_size: int = Field(
        default_factory=lambda: get_defaults()["dft"]["chunk_size"]
    )
    default_dft_max_workers: int = Field(
        default_factory=lambda: get_defaults()["dft"]["max_workers"]
    )

    # MACE Defaults
    default_mace_model_path: str = Field(
        default_factory=lambda: get_defaults()["mace_model_path"]
    )
    default_mace_device: str = Field(
        default_factory=lambda: get_defaults()["mace_device"]
    )
    default_mace_dtype: str = Field(
        default_factory=lambda: get_defaults()["mace_dtype"]
    )
    default_mace_batch_size: int = Field(
        default_factory=lambda: get_defaults()["mace_batch_size"]
    )

    # Trainer File Defaults
    default_trainer_baseline_suffix: str = Field(
        default_factory=lambda: get_defaults()["trainer_delta_baseline_suffix"]
    )
    default_trainer_mock_potential_name: str = Field(
        default_factory=lambda: get_defaults()["trainer_mock_potential_name"]
    )

    # Pacemaker Defaults
    pacemaker_default_potential_filename: str = Field(
        default_factory=lambda: get_defaults()["pacemaker_default_potential_filename"]
    )
    pacemaker_default_embeddings: dict[str, Any] = Field(
        default_factory=lambda: get_defaults()["pacemaker_default_embeddings"]
    )
    pacemaker_default_kappa: float = Field(
        default=0.3, description="Default kappa for Pacemaker loss"
    )
    pacemaker_default_seed: int = Field(
        default=42, description="Default random seed for Pacemaker"
    )
    pacemaker_default_delta_spline_bins: float = Field(
        default=0.001, description="Default delta spline bins for Pacemaker"
    )
    pacemaker_default_evaluator: str = Field(
        default="tensorpot", description="Default evaluator for Pacemaker"
    )
    pacemaker_default_display_step: int = Field(
        default=100, description="Default display step for Pacemaker"
    )
    pacemaker_default_patience: int = Field(
        default=50, description="Default patience for Pacemaker"
    )
    pacemaker_default_w_energy: float = Field(
        default=1.0, description="Default energy weight"
    )
    pacemaker_default_w_forces: float = Field(
        default=1.0, description="Default forces weight"
    )
    pacemaker_default_w_stress: float = Field(
        default=0.1, description="Default stress weight"
    )

    # DFT Defaults
    default_dft_ecutwfc: float = Field(
        default_factory=lambda: get_defaults()["dft"]["ecutwfc"]
    )
    default_dft_ecutrho: float = Field(
        default_factory=lambda: get_defaults()["dft"]["ecutrho"]
    )
    default_dft_conv_thr: float = Field(
        default_factory=lambda: get_defaults()["dft"]["conv_thr"]
    )
    default_dft_occupations: str = Field(
        default_factory=lambda: get_defaults()["dft"]["occupations"]
    )
    default_dft_smearing_method: str = Field(
        default_factory=lambda: get_defaults()["dft"]["smearing_method"]
    )

    dft_recoverable_errors: list[str] = Field(
        default_factory=lambda: get_defaults()["dft"]["recoverable_errors"]
    )
    dft_allowed_input_sections: list[str] = Field(
        default_factory=lambda: get_defaults()["dft"]["allowed_input_sections"]
    )

    allowed_potential_paths: list[str] = Field(
        default_factory=lambda: get_defaults()["allowed_potential_paths"]
    )

    allowed_feature_keys: list[str] = Field(
        default_factory=lambda: get_defaults()["allowed_feature_keys"]
    )

    default_dynamics_gamma_threshold: float = Field(
        default_factory=lambda: get_defaults()["dynamics_gamma_threshold"]
    )

    default_dynamics_templates: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "zbl": [
                "pair_style hybrid/overlay pace zbl 4.0 5.0",
                "pair_coeff * * pace {path} {elements}",
                "pair_coeff * * zbl 0.0 0.0",
            ],
            "lj": [
                "pair_style hybrid/overlay pace lj/cut 10.0",
                "pair_coeff * * pace {path} {elements}",
                "pair_coeff * * lj/cut 1.0 1.0",
            ],
            "default": [
                "pair_style pace",
                "pair_coeff * * pace {path} {elements}",
            ],
        }
    )

    PICKLE_SECURITY_WARNING: str = Field(
        default_factory=lambda: get_defaults()["pickle_security_warning"]
    )

    valid_key_regex: str = Field(
        default_factory=lambda: get_defaults()["valid_key_regex"]
    )
    valid_value_regex: str = Field(
        default_factory=lambda: get_defaults()["valid_value_regex"]
    )
    mace_param_key_regex: str = Field(
        default_factory=lambda: get_defaults()["mace_param_key_regex"]
    )
    mace_param_value_regex: str = Field(
        default_factory=lambda: get_defaults()["mace_param_value_regex"]
    )

    TRAINER_TEMP_PREFIX_TRAIN: str = Field(
        default_factory=lambda: get_defaults()["temp_prefix_train"]
    )
    TRAINER_TEMP_PREFIX_ACTIVE: str = Field(
        default_factory=lambda: get_defaults()["temp_prefix_active"]
    )

    DFT_TEMP_PREFIX: str = Field(
        default_factory=lambda: get_defaults()["dft_temp_prefix"]
    )

    max_energy_ev: float = Field(
        default_factory=lambda: get_defaults()["max_energy_ev"]
    )
    max_force_ev_a: float = Field(
        default_factory=lambda: get_defaults()["max_force_ev_a"]
    )
    composition_tolerance: float = Field(
        default_factory=lambda: get_defaults()["composition_tolerance"]
    )

    physics_phonon_supercell: list[int] = Field(
        default_factory=lambda: get_defaults()["physics_phonon_supercell"]
    )
    physics_phonon_tolerance: float = Field(
        default_factory=lambda: get_defaults()["physics_phonon_tolerance"]
    )
    physics_eos_strain: float = Field(
        default_factory=lambda: get_defaults()["physics_eos_strain"]
    )
    physics_eos_points: int = Field(
        default_factory=lambda: get_defaults()["physics_eos_points"]
    )
    physics_elastic_strain: float = Field(
        default_factory=lambda: get_defaults()["physics_elastic_strain"]
    )

    max_atoms_dft: int = Field(
        default_factory=lambda: get_defaults()["max_atoms_dft"]
    )
    dynamics_halt_probability: float = Field(
        default_factory=lambda: get_defaults()["dynamics_halt_probability"]
    )

    default_dataset_file: str = Field(
        default_factory=lambda: get_defaults()["default_dataset_file"]
    )
    default_validation_file: str = Field(
        default_factory=lambda: get_defaults()["default_validation_file"]
    )
    default_training_file: str = Field(
        default_factory=lambda: get_defaults()["default_training_file"]
    )
    internal_base_potential_version: str = Field(
        default="0.0.0",
        description="Version string for internal base potentials used in workflows"
    )
    default_candidates_file: str = Field(
        default_factory=lambda: get_defaults()["default_candidates_file"]
    )
    default_selected_file: str = Field(
        default_factory=lambda: get_defaults()["default_selected_file"]
    )
    dataset_extension: str = Field(
        default_factory=lambda: get_defaults()["dataset_extension"]
    )

    # Generator Defaults
    direct_oversample: int = Field(
        default_factory=lambda: get_defaults()["structure_direct_oversample"],
        description="Oversampling factor for Direct Sampling"
    )
    direct_batch_size: int = Field(
        default_factory=lambda: get_defaults()["structure_direct_batch_size"],
        description="Batch size for Direct Sampling"
    )
    direct_box_size: float = Field(
        default_factory=lambda: get_defaults()["structure_direct_box_size"],
        description="Box size for Direct Sampling"
    )

    # MACE Defaults
    mace_default_model_name: str = Field(
        default="mace_model_compiled.model", description="Default MACE model filename"
    )
    mace_default_max_epochs: int = Field(
        default_factory=lambda: get_defaults()["mace_default_max_epochs"],
        description="Default max epochs for MACE training"
    )

    mace_allowed_train_params: frozenset[str] = Field(
        default_factory=lambda: frozenset({
            "model", "train_file", "valid_file", "test_file", "E0s", "config", "seed",
            "device", "batch_size", "max_num_epochs", "patience", "eval_interval",
            "keep_checkpoints", "restart_latest", "loss", "ems", "forces_weight",
            "energy_weight", "stress_weight", "virial_weight", "lr", "scheduler",
            "decay", "clip_grad", "swa", "start_swa", "swa_lr", "swa_forces_weight",
            "swa_energy_weight", "swa_stress_weight", "swa_virial_weight", "r_max",
            "num_radial_basis", "num_cutoff_basis", "interaction", "interaction_first",
            "max_ell", "correlation", "hidden_irreps", "MLP_irreps", "gate",
            "scaling", "avg_num_neighbors", "compute_avg_num_neighbors",
            "compute_stress", "compute_forces", "compute_virial", "error_table",
            "default_dtype", "checkpoints_dir", "log_dir", "name", "wandb_name",
            "wandb_project", "wandb_entity", "wandb_log_hypers", "foundation_model",
            "finetune", "distributed",
        })
    )

    # Oracle Defaults
    oracle_chunk_size: int = Field(
        default_factory=lambda: get_defaults()["oracle_chunk_size"],
        description="Chunk size for Oracle batch processing"
    )

    # Workflow Defaults (Added for refactoring)
    workflow_log_interval: int = Field(
        default=100, description="Logging interval for structure processing"
    )
    workflow_default_seeds: int = Field(
        default=5, description="Default number of seeds for exploration"
    )

    @field_validator("max_config_size")
    @classmethod
    def validate_max_config_size(cls, v: int) -> int:
        if v < 1024:  # Minimum 1KB
            msg = "max_config_size must be at least 1KB"
            raise ValueError(msg)
        return v

    @field_validator("dynamics_halt_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability value."""
        return _validate_probability(v)

    @field_validator("valid_key_regex", "valid_value_regex")
    @classmethod
    def validate_regex(cls, v: str) -> str:
        """Validate regex pattern."""
        return _validate_regex(v)


def _validate_probability(v: float) -> float:
    """Reusable probability validation."""
    if not 0.0 <= v <= 1.0:
        msg = "Probability must be between 0.0 and 1.0"
        raise ValueError(msg)
    return v


def _validate_regex(v: str) -> str:
    """Reusable regex validation."""
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
        return validate_parameters(v)


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the project")
    root_dir: Path = Field(..., description="Root directory of the project")

    @field_validator("root_dir")
    @classmethod
    def validate_root_dir(cls, v: Path) -> Path:
        """Validate root directory."""
        # Use centralized safe path validation
        try:
            # validate_safe_path handles skip_file_checks internally where appropriate
            validate_safe_path(v)
        except ValueError as e:
            msg = f"Invalid root directory: {e}"
            raise ValueError(msg) from e

        # Resolve to absolute
        return v.resolve() if v.exists() else v.absolute()


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
            # validate_safe_path performs resolution and whitelist checking.
            # We return the resolved path string to ensure downstream usage is safe.
            safe_path = validate_safe_path(path)
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
        default_factory=lambda: get_defaults()["dft"]["embedding_enabled"],
        description="Enable periodic embedding for non-periodic structures"
    )
    embedding_buffer: float = Field(
        default_factory=lambda: get_defaults()["dft"]["embedding_buffer"],
        description="Buffer size for periodic embedding (Angstrom)"
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
                validate_safe_path(path)
            except ValueError as e:
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
        return validate_parameters(v)


class OracleConfig(BaseModel):
    """Oracle module configuration."""

    model_config = ConfigDict(extra="forbid")

    dft: DFTConfig = Field(..., description="DFT configuration")
    mace: MaceConfig | None = Field(default=None, description="MACE configuration")
    mock: bool = Field(
        default_factory=lambda: get_defaults()["oracle_mock"],
        description="Use mock oracle for testing"
    )


class StructureGeneratorConfig(BaseModuleConfig):
    """Structure Generator module configuration."""

    strategy: str = Field(
        default_factory=lambda: get_defaults()["structure_strategy"],
        description="Generation strategy (e.g., 'random', 'adaptive')",
    )
    initial_exploration: str = Field(
        default_factory=lambda: get_defaults()["structure_initial_exploration"],
        description="Initial exploration strategy (m3gnet or random)",
    )
    strain_range: float = Field(
        default_factory=lambda: get_defaults()["structure_strain_range"],
        ge=0.0,
        description="Maximum strain range for random perturbation",
    )
    rattle_amplitude: float = Field(
        default_factory=lambda: get_defaults()["structure_rattle_amplitude"],
        ge=0.0,
        description="Standard deviation for atomic rattling (Angstrom)",
    )
    defect_density: float = Field(
        default_factory=lambda: get_defaults()["structure_defect_density"],
        ge=0.0,
        description="Target defect density for defect strategy",
    )


class TrainerConfig(BaseModuleConfig):
    """Trainer module configuration."""

    potential_type: str = Field(
        default=CONSTANTS.default_trainer_potential, description="Type of potential to train"
    )
    mock: bool = Field(
        default_factory=lambda: get_defaults()["trainer_mock"],
        description="Use mock trainer for testing"
    )
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

    executable: str = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_executable"],
        description="Path to EON executable",
    )
    parameters: dict[str, Any] = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_parameters"],
        description="EON parameters",
    )
    mock: bool = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_mock"],
        description="Use mock EON",
    )
    log_file: str = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_log_file"],
        description="EON log file",
    )
    max_steps: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_steps"],
        description="Max steps",
    )
    temperature: float = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_temperature"],
        description="Temperature",
    )
    pressure: float = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_pressure"],
        description="Pressure",
    )
    seed: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_seed"],
        description="Random seed",
    )
    output_dir: str = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_output_dir"],
        description="Output directory",
    )
    restart_file: str = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_restart_file"],
        description="Restart file",
    )
    checkpoint_interval: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_checkpoint_interval"],
        description="Checkpoint interval",
    )
    max_restarts: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_restarts"],
        description="Max restarts",
    )
    max_failures: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_failures"],
        description="Max failures",
    )
    max_time: float = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_time"],
        description="Max time",
    )
    max_memory: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_memory"],
        description="Max memory",
    )
    max_disk: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_disk"],
        description="Max disk",
    )
    max_network: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_network"],
        description="Max network",
    )
    max_cpu: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_cpu"],
        description="Max cpu",
    )
    max_gpu: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_gpu"],
        description="Max gpu",
    )
    max_threads: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_threads"],
        description="Max threads",
    )
    max_processes: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_processes"],
        description="Max processes",
    )
    max_queues: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_queues"],
        description="Max queues",
    )
    max_jobs: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_jobs"],
        description="Max jobs",
    )
    max_tasks: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_tasks"],
        description="Max tasks",
    )
    max_workers: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_workers"],
        description="Max workers",
    )
    max_concurrent: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_concurrent"],
        description="Max concurrent",
    )
    max_parallel: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_parallel"],
        description="Max parallel",
    )
    max_batch: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_batch"],
        description="Max batch",
    )
    max_chunk: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_chunk"],
        description="Max chunk",
    )
    max_buffer: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_buffer"],
        description="Max buffer",
    )
    max_cache: int = Field(
        default_factory=lambda: get_defaults()["dynamics_eon_max_cache"],
        description="Max cache",
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters dictionary."""
        return validate_parameters(v)


class DynamicsEngineConfig(BaseModuleConfig):
    """Dynamics Engine module configuration."""

    engine: str = Field(default=CONSTANTS.default_engine, description="MD/kMC engine")
    gamma_threshold: float = Field(
        default_factory=lambda: get_defaults()["dynamics_gamma_threshold"],
        description="Threshold for extrapolation grade (gamma) to trigger halt",
    )
    timestep: float = Field(
        default_factory=lambda: get_defaults()["dynamics_timestep"],
        description="Timestep in ps",
    )
    temperature: float = Field(
        default_factory=lambda: get_defaults()["dynamics_temperature"],
        description="Temperature in K",
    )
    pressure: float = Field(
        default_factory=lambda: get_defaults()["dynamics_pressure"],
        description="Pressure in Bar",
    )
    n_steps: int = Field(
        default_factory=lambda: get_defaults()["dynamics_n_steps"],
        description="Number of MD steps",
    )
    hybrid_baseline: str = Field(
        default_factory=lambda: get_defaults()["dynamics_hybrid_baseline"],
        description="Hybrid potential baseline (zbl or lj)",
    )
    mock: bool = Field(
        default_factory=lambda: get_defaults()["dynamics_mock"],
        description="Use mock engine for testing",
    )
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


class Step1DirectSamplingConfig(BaseModel):
    """Configuration for Step 1: Direct Sampling."""

    model_config = ConfigDict(extra="forbid")
    target_points: int = Field(
        default_factory=lambda: get_defaults()["step1_target_points"],
        ge=1,
        description="Number of target points",
    )
    objective: str = Field(
        default_factory=lambda: get_defaults()["step1_objective"],
        description="Optimization objective",
    )


class Step2ActiveLearningConfig(BaseModel):
    """Configuration for Step 2: Active Learning."""

    model_config = ConfigDict(extra="forbid")
    uncertainty_threshold: float = Field(
        default_factory=lambda: get_defaults()["step2_uncertainty_threshold"],
        ge=0.0,
        description="Uncertainty threshold",
    )
    dft_calculator: str = Field(
        default_factory=lambda: get_defaults()["step2_dft_calculator"],
        description="DFT calculator name",
    )
    cycles: int = Field(
        default_factory=lambda: get_defaults()["step2_cycles"],
        ge=1,
        description="Number of active learning cycles",
    )
    n_select: int = Field(
        default_factory=lambda: get_defaults()["step2_n_select"],
        ge=1,
        description="Number of structures to select per cycle",
    )


class Step3MaceFinetuneConfig(BaseModel):
    """Configuration for Step 3: MACE Fine-tuning."""

    model_config = ConfigDict(extra="forbid")
    base_model: str = Field(
        default_factory=lambda: get_defaults()["step3_base_model"],
        description="Base MACE model",
    )
    epochs: int = Field(
        default_factory=lambda: get_defaults()["step3_epochs"],
        ge=1,
        description="Number of epochs"
    )


class Step4SurrogateSamplingConfig(BaseModel):
    """Configuration for Step 4: Surrogate Sampling."""

    model_config = ConfigDict(extra="forbid")
    target_points: int = Field(
        default_factory=lambda: get_defaults()["step4_target_points"],
        ge=1,
        description="Number of target points",
    )
    method: str = Field(
        default_factory=lambda: get_defaults()["step4_method"], description="Sampling method"
    )


class Step7PacemakerFinetuneConfig(BaseModel):
    """Configuration for Step 7: Pacemaker Fine-tuning."""

    model_config = ConfigDict(extra="forbid")
    enable: bool = Field(
        default_factory=lambda: get_defaults()["step7_enable"],
        description="Enable Delta Learning",
    )
    weight_dft: float = Field(
        default_factory=lambda: get_defaults()["step7_weight_dft"],
        ge=0.0,
        description="Weight for DFT data",
    )


class DistillationConfig(BaseModel):
    """Configuration for MACE Distillation Workflow."""

    model_config = ConfigDict(extra="forbid")
    enable_mace_distillation: bool = Field(
        default_factory=lambda: get_defaults()["distillation_enable"],
        description="Enable MACE distillation"
    )

    # File paths for intermediate artifacts
    pool_file: str = Field(
        default_factory=lambda: get_defaults()["distillation"]["pool_file"],
        description="Filename for the initial structure pool"
    )
    surrogate_file: str = Field(
        default_factory=lambda: get_defaults()["distillation"]["surrogate_file"],
        description="Filename for generated surrogate structures"
    )
    surrogate_dataset_file: str = Field(
        default_factory=lambda: get_defaults()["distillation"]["surrogate_dataset_file"],
        description="Filename for labeled surrogate dataset"
    )

    step1_direct_sampling: Step1DirectSamplingConfig = Field(
        default_factory=Step1DirectSamplingConfig
    )
    step2_active_learning: Step2ActiveLearningConfig = Field(
        default_factory=Step2ActiveLearningConfig
    )
    step3_mace_finetune: Step3MaceFinetuneConfig = Field(default_factory=Step3MaceFinetuneConfig)
    step4_surrogate_sampling: Step4SurrogateSamplingConfig = Field(
        default_factory=Step4SurrogateSamplingConfig
    )
    step7_pacemaker_finetune: Step7PacemakerFinetuneConfig = Field(
        default_factory=Step7PacemakerFinetuneConfig
    )

    # Workflow Performance Settings
    batch_size: int = Field(
        default=100,
        ge=1,
        description="General batch size for processing structures"
    )
    write_buffer_size: int = Field(
        default=1000,
        ge=1,
        description="Buffer size for writing labeled structures to disk"
    )

    @field_validator("pool_file")
    @classmethod
    def validate_pool_file(cls, v: str) -> str:
        if not v or not v.strip():
            msg = "pool_file cannot be empty"
            raise ValueError(msg)
        return v


class ValidatorConfig(BaseModel):
    """Validator module configuration."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default_factory=lambda: CONSTANTS.default_validator_metrics,
        description="Metrics to validate",
    )
    thresholds: dict[str, float] = Field(default_factory=dict, description="Validation thresholds")
    test_set_ratio: float = Field(
        default_factory=lambda: get_defaults()["validator_test_ratio"],
        description="Ratio of dataset to use for testing"
    )
    phonon_supercell: list[int] = Field(
        default_factory=lambda: CONSTANTS.physics_phonon_supercell,
        description="Supercell for phonon calculation",
    )
    eos_strain: float = Field(
        default=CONSTANTS.physics_eos_strain, description="Strain range for EOS calculation"
    )
    eos_points: int = Field(
        default=CONSTANTS.physics_eos_points, description="Number of points for EOS calculation"
    )
    elastic_strain: float = Field(
        default=CONSTANTS.physics_elastic_strain,
        description="Strain for elastic constants calculation",
    )
    phonon_tolerance: float = Field(
        default=CONSTANTS.physics_phonon_tolerance,
        description="Tolerance for phonon stability (min frequency)",
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
    validation_file: str = Field(
        default=CONSTANTS.default_validation_file,
        description="Filename for the validation dataset",
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
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version."""
        if not re.match(CONSTANTS.version_regex, v):
            msg = f"Invalid version format: {v}. Must match {CONSTANTS.version_regex}"
            raise ValueError(msg)
        return v


# Configuration loading logic moved to pyacemaker.core.config_loader
