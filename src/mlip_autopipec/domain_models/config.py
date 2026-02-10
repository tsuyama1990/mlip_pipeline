from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .enums import GeneratorType, OracleType, TrainerType


class ComponentConfig(BaseModel):
    """Base configuration for all components."""
    model_config = ConfigDict(extra="forbid")

# --- Generator Configurations ---
class MockGeneratorConfig(ComponentConfig):
    type: Literal[GeneratorType.MOCK] = GeneratorType.MOCK
    params: dict[str, Any] = Field(default_factory=dict)

class RandomGeneratorConfig(ComponentConfig):
    type: Literal[GeneratorType.RANDOM] = GeneratorType.RANDOM
    params: dict[str, Any] = Field(default_factory=dict)

class AdaptiveGeneratorConfig(ComponentConfig):
    type: Literal[GeneratorType.ADAPTIVE] = GeneratorType.ADAPTIVE
    params: dict[str, Any] = Field(default_factory=dict)

GeneratorConfig = Annotated[
    MockGeneratorConfig | RandomGeneratorConfig | AdaptiveGeneratorConfig,
    Field(discriminator="type")
]

# --- Oracle Configurations ---
class MockOracleConfig(ComponentConfig):
    type: Literal[OracleType.MOCK] = OracleType.MOCK
    params: dict[str, Any] = Field(default_factory=dict)

class QEOracleConfig(ComponentConfig):
    type: Literal[OracleType.QE] = OracleType.QE
    command: str

OracleConfig = Annotated[
    MockOracleConfig | QEOracleConfig,
    Field(discriminator="type")
]

# --- Trainer Configurations ---
class MockTrainerConfig(ComponentConfig):
    type: Literal[TrainerType.MOCK] = TrainerType.MOCK
    dataset_path: Path

class PaceTrainerConfig(ComponentConfig):
    type: Literal[TrainerType.PACE] = TrainerType.PACE
    dataset_path: Path

TrainerConfig = Annotated[
    MockTrainerConfig | PaceTrainerConfig,
    Field(discriminator="type")
]

# --- Orchestrator Configuration ---
class OrchestratorConfig(BaseModel):
    work_dir: Path
    max_iterations: int = 10
    model_config = ConfigDict(extra="forbid")

# --- Root Configuration ---
class Config(BaseModel):
    orchestrator: OrchestratorConfig
    generator: GeneratorConfig
    oracle: OracleConfig
    trainer: TrainerConfig

    model_config = ConfigDict(extra="forbid")
