from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExplorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "random", "md"] = "mock"
    n_structures: int = Field(default=2, ge=1)

class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "espresso"] = "mock"

class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "pacemaker"] = "mock"
    # Configurable output path for potential
    potential_output_name: str = "potential.yace"

    @field_validator("potential_output_name")
    @classmethod
    def check_potential_extension(cls, v: str) -> str:
        if not v.endswith(".yace"):
            msg = f"Potential output name must end with .yace, got {v}"
            raise ValueError(msg)
        return v

class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"

class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Field(default=Path("./_work"))
    max_cycles: int = Field(ge=1)
    random_seed: int = Field(default=42)
    max_accumulated_structures: int = Field(default=1000, ge=0)
    # Optional initial potential path
    initial_potential: Path | None = None

    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
