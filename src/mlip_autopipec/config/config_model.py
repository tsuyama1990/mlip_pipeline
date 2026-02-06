from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExplorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "random", "md"] = "mock"

class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "espresso"] = "mock"

class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "pacemaker"] = "mock"
    # Configurable output path for potential
    potential_output_name: str = "potential.yace"

class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"

class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Field(default=Path("./_work"))
    max_cycles: int = Field(ge=1)
    random_seed: int = Field(default=42)
    # Optional initial potential path
    initial_potential: Path | None = None

    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
