from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class ExplorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["random", "md", "mock"] = Field("mock", description="Type of exploration engine")

class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["espresso", "mock"] = Field("mock", description="Type of oracle")

class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["pacemaker", "mock"] = Field("mock", description="Type of trainer")

class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Field(..., description="Directory to store results")
    max_cycles: int = Field(..., gt=0, description="Number of active learning iterations")
    random_seed: int = Field(42, description="Random seed for reproducibility")

    explorer: ExplorerConfig = Field(default_factory=lambda: ExplorerConfig())
    oracle: OracleConfig = Field(default_factory=lambda: OracleConfig())
    trainer: TrainerConfig = Field(default_factory=lambda: TrainerConfig())
