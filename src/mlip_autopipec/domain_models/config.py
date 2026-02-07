from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Audit: Validate params, required with default factory
    params: dict[str, Any] = Field(default_factory=dict)

class OracleConfig(BaseConfig):
    type: Literal["mock", "qe", "vasp"]

class TrainerConfig(BaseConfig):
    type: Literal["mock", "pacemaker"]

class DynamicsConfig(BaseConfig):
    type: Literal["mock", "lammps"]

class GeneratorConfig(BaseConfig):
    type: Literal["mock", "random"]

class ValidatorConfig(BaseConfig):
    type: Literal["mock", "standard"]

class SelectorConfig(BaseConfig):
    type: Literal["mock", "random", "d_opt"]

class GlobalConfig(BaseModel):
    """
    Root configuration object for the pipeline.
    """
    model_config = ConfigDict(extra="forbid")

    project_name: str
    seed: int
    workdir: Path = Path("mlip_run")

    # Audit: Add max_cycles and initial_structure_path
    max_cycles: int = 5
    initial_structure_path: Path | None = None

    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    generator: GeneratorConfig
    validator: ValidatorConfig
    selector: SelectorConfig
