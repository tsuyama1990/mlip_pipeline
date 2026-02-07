from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "qe"]
    params: dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "mace", "allegro"]
    params: dict[str, Any] = Field(default_factory=dict)


class DynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "lammps"]
    params: dict[str, Any] = Field(default_factory=dict)


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "rattle"]
    params: dict[str, Any] = Field(default_factory=dict)


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "lammps"]
    params: dict[str, Any] = Field(default_factory=dict)


class SelectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "greedy"]
    params: dict[str, Any] = Field(default_factory=dict)


class GlobalConfig(BaseModel):
    """
    The root configuration object.
    """

    model_config = ConfigDict(extra="forbid")

    workdir: Path
    max_cycles: int
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    generator: GeneratorConfig
    validator: ValidatorConfig
    selector: SelectorConfig
