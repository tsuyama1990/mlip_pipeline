from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.constants import DEFAULT_WORK_DIR, MAX_CYCLES
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


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


class TrainerConfig(BaseComponentConfig):
    type: TrainerType = TrainerType.MOCK
    training_set_path: str | None = None
    validation_split: float = 0.1


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

        with path.open("r") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                msg = f"Invalid YAML file: {e}"
                raise ValueError(msg) from e

        return cls.model_validate(data)
