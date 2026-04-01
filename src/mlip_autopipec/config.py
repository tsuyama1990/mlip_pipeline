import os
import urllib.parse
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


def validate_safe_path(v: str | Path | None) -> Path | None:
    """
    Validator to ensure paths are safe:
    - Must not traverse up (..) from the base resolution point.
    - Must not be absolute paths pointing to sensitive system directories.
    - Resolves the path to check canonical form.
    - Handles URL encoded paths.
    """
    if v is None:
        return None

    path_str = str(v)
    # Decode URL encoded characters
    path_str = urllib.parse.unquote(path_str)
    path = Path(path_str)

    # Check for simple traversal before resolution
    if ".." in path_str or ".." in path.parts:
        msg = f"Path traversal detected: {path}"
        raise ValueError(msg)

    # Resolve to absolute path
    # If path is relative, it resolves against CWD
    try:
        resolved_path = path.resolve()
    except OSError:
        # Path might not exist, but we can still resolve logically if possible
        # Or if it fails, it might be invalid chars.
        resolved_path = path.absolute()

    # Define sensitive roots we absolutely want to avoid writing to/reading from blindly
    sensitive_roots = [
        Path("/etc"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/var"),
        Path("/boot"),
        Path("/sys"),
        Path("/proc"),
        Path("/root"),
    ]

    resolved_path_str = str(resolved_path)
    for root in sensitive_roots:
        root_str = str(root)
        # Using string comparison for stricter "starts with" check
        if resolved_path_str == root_str or resolved_path_str.startswith(f"{root_str}/"):
            _raise_unsafe_path_error(resolved_path)

    return path


def _raise_unsafe_path_error(path: Path) -> None:
    """Helper to raise ValueError for unsafe paths."""
    msg = f"Path points to restricted system directory: {path}"
    raise ValueError(msg)


class BaseComponentConfig(BaseModel):
    """Base configuration for all components."""

    model_config = ConfigDict(extra="forbid")


class GeneratorConfig(BaseComponentConfig):
    type: GeneratorType = GeneratorType.RANDOM
    seed: int = 42
    n_structures: int = 10


class OracleConfig(BaseComponentConfig):
    type: OracleType = OracleType.MOCK
    scf_kspacing: float = 0.5
    scf_flags: dict[str, str] = Field(default_factory=dict)
    pseudopotentials: dict[str, str] = Field(default_factory=dict)

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudos(cls, v: dict[str, str]) -> dict[str, str]:
        for path in v.values():
            validate_safe_path(path)
        return v


class TrainerConfig(BaseComponentConfig):
    type: TrainerType = TrainerType.MOCK
    training_set_path: Path | None = None
    validation_split: float = 0.1

    @field_validator("training_set_path")
    @classmethod
    def check_safe_path(cls, v: Path | None) -> Path | None:
        return validate_safe_path(v)


class DynamicsConfig(BaseComponentConfig):
    type: DynamicsType = DynamicsType.MOCK
    temperature: float = 300.0
    timestep: float = 0.001
    n_steps: int = 1000


class ValidatorConfig(BaseComponentConfig):
    type: ValidatorType = ValidatorType.MOCK
    threshold: float = 0.01


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator loop."""

    model_config = ConfigDict(extra="forbid")

    # Defaults defined here directly using ENV vars
    work_dir: Path = Field(default_factory=lambda: Path(os.getenv("MLIP_WORK_DIR", "work")))
    max_cycles: int = Field(default_factory=lambda: int(os.getenv("MLIP_MAX_CYCLES", "10")), ge=1)
    continue_from_cycle: int = 0
    stop_on_failure: bool = True

    @field_validator("work_dir")
    @classmethod
    def check_safe_path(cls, v: Path) -> Path:
        validate_safe_path(v)
        return v


class ExperimentConfig(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid")

    orchestrator: OrchestratorConfig = Field(default_factory=lambda: OrchestratorConfig())
    generator: GeneratorConfig = Field(default_factory=lambda: GeneratorConfig())
    oracle: OracleConfig = Field(default_factory=lambda: OracleConfig())
    trainer: TrainerConfig = Field(default_factory=lambda: TrainerConfig())
    dynamics: DynamicsConfig = Field(default_factory=lambda: DynamicsConfig())
    validator: ValidatorConfig = Field(default_factory=lambda: ValidatorConfig())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        # Security: Safe Load is already used, but let's be explicit
        with path.open("r") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                msg = f"Invalid YAML file: {e}"
                raise ValueError(msg) from e

        return cls.model_validate(data)
