from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    params: dict[str, Any] | None = None

class OracleConfig(BaseConfig):
    type: Literal["mock", "qe", "vasp"]

class TrainerConfig(BaseConfig):
    type: Literal["mock", "pacemaker"]

class DynamicsConfig(BaseConfig):
    type: Literal["mock", "lammps"]

class GeneratorConfig(BaseConfig):
    type: Literal["mock", "random"]

class GlobalConfig(BaseModel):
    """
    Root configuration object for the pipeline.
    """
    model_config = ConfigDict(extra="forbid")

    project_name: str
    seed: int
    workdir: Path = Path("mlip_run")
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    generator: GeneratorConfig
