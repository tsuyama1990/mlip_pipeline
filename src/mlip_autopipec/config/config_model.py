from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OracleConfig(BaseModel):
    """
    Configuration for the Oracle component (DFT/Ground Truth).
    """
    type: Literal["mock", "espresso"] = Field(default="mock", description="Type of Oracle to use")

    model_config = ConfigDict(extra="forbid")

class TrainerConfig(BaseModel):
    """
    Configuration for the Trainer component (MLIP fitting).
    """
    type: Literal["mock", "pacemaker"] = Field(default="mock", description="Type of Trainer to use")

    model_config = ConfigDict(extra="forbid")

class ExplorerConfig(BaseModel):
    """
    Configuration for the Explorer component (Structure Generation/MD).
    """
    type: Literal["mock", "adaptive"] = Field(default="mock", description="Type of Explorer to use")

    model_config = ConfigDict(extra="forbid")

class GlobalConfig(BaseModel):
    """
    Root configuration for the MLIP system.
    """
    work_dir: Path = Field(description="Working directory for all outputs")
    max_cycles: int = Field(default=10, ge=1, description="Maximum number of Active Learning cycles")

    oracle: OracleConfig = Field(default_factory=OracleConfig, description="Oracle configuration")
    trainer: TrainerConfig = Field(default_factory=TrainerConfig, description="Trainer configuration")
    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig, description="Explorer configuration")

    model_config = ConfigDict(extra="forbid")
