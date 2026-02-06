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
    type: str = "mock"

class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Required fields (no defaults) per audit requirements
    work_dir: Path
    random_seed: int

    max_cycles: int = Field(ge=1)

    # Optional limits
    max_accumulated_structures: int | None = None

    # Optional initial potential path
    initial_potential: Path | None = None

    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
