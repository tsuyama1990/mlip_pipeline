from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.constants import DEFAULT_WORK_DIR, MAX_CYCLES
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


def validate_safe_path(v: str | Path | None) -> Path | None:
    """
    Validator to ensure paths are safe and do not traverse up (..).
    """
    if v is None:
        return None

    path = Path(v)
    if ".." in str(path) or path.is_absolute():
        # Ideally we allow absolute paths if they are within allowed directories,
        # but for now let's just warn or restrict relative traversal.
        # The audit requirement specifically mentioned path traversal.
        pass

    # Check for suspicious traversal
    # Resolve path to check if it tries to escape
    # This is tricky without a base root.
    # We will just forbid '..' components for now as a strict rule for user inputs.
    if ".." in path.parts:
        msg = f"Path traversal detected: {path}"
        raise ValueError(msg)

    return path


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
            if ".." in Path(path).parts:
                msg = f"Path traversal detected in pseudopotential: {path}"
                raise ValueError(msg)
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

    work_dir: Path = DEFAULT_WORK_DIR
    max_cycles: int = Field(default=MAX_CYCLES, ge=1)
    continue_from_cycle: int = 0
    stop_on_failure: bool = True

    @field_validator("work_dir")
    @classmethod
    def check_safe_path(cls, v: Path) -> Path:
        # Work dir must be safe
        if ".." in v.parts:
            msg = f"Path traversal detected in work_dir: {v}"
            raise ValueError(msg)
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
