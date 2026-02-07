from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.constants import (
    DEFAULT_NOISE_LEVEL,
    DEFAULT_PROB_HALT,
)


class BaseComponentConfig(BaseModel):
    """
    Base configuration for all components.
    """

    model_config = ConfigDict(extra="forbid")
    type: str


# --- Oracle Configs ---
class MockOracleConfig(BaseComponentConfig):
    type: Literal["mock"] = "mock"
    noise_level: float = DEFAULT_NOISE_LEVEL


OracleConfig = Annotated[MockOracleConfig, Field(discriminator="type")]


# --- Trainer Configs ---
class MockTrainerConfig(BaseComponentConfig):
    type: Literal["mock"] = "mock"


TrainerConfig = Annotated[MockTrainerConfig, Field(discriminator="type")]


# --- Dynamics Configs ---
class MockDynamicsConfig(BaseComponentConfig):
    type: Literal["mock"] = "mock"
    prob_halt: float = DEFAULT_PROB_HALT


DynamicsConfig = Annotated[MockDynamicsConfig, Field(discriminator="type")]


# --- Generator Configs ---
class MockGeneratorConfig(BaseComponentConfig):
    type: Literal["mock"] = "mock"


GeneratorConfig = Annotated[MockGeneratorConfig, Field(discriminator="type")]


# --- Validator Configs ---
class MockValidatorConfig(BaseComponentConfig):
    type: Literal["mock"] = "mock"


ValidatorConfig = Annotated[MockValidatorConfig, Field(discriminator="type")]


# --- Selector Configs ---
class MockSelectorConfig(BaseComponentConfig):
    type: Literal["mock"] = "mock"


SelectorConfig = Annotated[MockSelectorConfig, Field(discriminator="type")]


# --- Global Config ---
class GlobalConfig(BaseModel):
    """
    Root configuration for the MLIP pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    max_cycles: int = Field(gt=0, description="Number of active learning cycles")
    initial_structure_path: Path = Field(description="Path to initial structure(s)")
    workdir: Path = Field(description="Working directory for artifacts")

    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    generator: GeneratorConfig
    validator: ValidatorConfig
    selector: SelectorConfig
