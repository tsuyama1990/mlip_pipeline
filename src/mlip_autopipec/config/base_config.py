from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class OracleConfig(BaseModel):
    type: Literal["mock", "quantum_espresso"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class TrainerConfig(BaseModel):
    type: Literal["mock", "pacemaker"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class DynamicsConfig(BaseModel):
    type: Literal["mock", "lammps", "eon"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class StructureGeneratorConfig(BaseModel):
    type: Literal["mock", "random"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ValidatorConfig(BaseModel):
    type: Literal["mock", "lammps_md"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class OrchestratorConfig(BaseModel):
    max_iterations: int = 10
    model_config = ConfigDict(extra="forbid")


class GlobalConfig(BaseModel):
    project_name: str
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    structure_generator: StructureGeneratorConfig = Field(
        default_factory=StructureGeneratorConfig
    )
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GlobalConfig":
        with Path(path).open("r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
