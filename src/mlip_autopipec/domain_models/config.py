from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseConfig(BaseModel):
    """
    Base configuration for modular components.
    """
    model_config = ConfigDict(extra="forbid")
    # Audit: Validate params, required with default factory
    params: dict[str, Any] = Field(default_factory=dict)

class OracleConfig(BaseConfig):
    """
    Configuration for the Oracle (ground truth calculator).
    """
    type: Literal["mock", "qe", "vasp"]

class TrainerConfig(BaseConfig):
    """
    Configuration for the Trainer (potential fitting).
    """
    type: Literal["mock", "pacemaker"]

class DynamicsConfig(BaseConfig):
    """
    Configuration for the Dynamics engine (MD/exploration).
    """
    type: Literal["mock", "lammps"]

class GeneratorConfig(BaseConfig):
    """
    Configuration for the Structure Generator.
    """
    type: Literal["mock", "random"]

class ValidatorConfig(BaseConfig):
    """
    Configuration for the Validator (quality assurance).
    """
    type: Literal["mock", "standard"]

class SelectorConfig(BaseConfig):
    """
    Configuration for the Selector (active learning strategy).
    """
    type: Literal["mock", "random", "d_opt"]

class GlobalConfig(BaseModel):
    """
    Root configuration object for the pipeline.
    """
    model_config = ConfigDict(extra="forbid")

    project_name: str
    seed: int
    # Workdir is now required (no default) to force explicit configuration
    workdir: Path

    # Audit: Add max_cycles and initial_structure_path
    max_cycles: int = 5
    initial_structure_path: Path | None = None

    # Dataset configuration
    dataset_maxlen: int = 1000

    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    generator: GeneratorConfig
    validator: ValidatorConfig
    selector: SelectorConfig

    @field_validator("max_cycles")
    @classmethod
    def validate_max_cycles(cls, v: int) -> int:
        if v < 1:
            msg = "max_cycles must be at least 1"
            raise ValueError(msg)
        return v

    @field_validator("dataset_maxlen")
    @classmethod
    def validate_dataset_maxlen(cls, v: int) -> int:
        if v < 1:
            msg = "dataset_maxlen must be at least 1"
            raise ValueError(msg)
        return v
