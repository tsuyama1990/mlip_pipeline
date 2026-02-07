from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Oracle Configs
class MockOracleConfig(BaseConfig):
    type: Literal["mock"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)


OracleConfig = Annotated[MockOracleConfig, Field(discriminator="type")]


# Trainer Configs
class MockTrainerConfig(BaseConfig):
    type: Literal["mock"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)


TrainerConfig = Annotated[MockTrainerConfig, Field(discriminator="type")]


# Dynamics Configs
class MockDynamicsConfig(BaseConfig):
    type: Literal["mock"] = "mock"
    halt_probability: float = 0.5
    params: dict[str, Any] = Field(default_factory=dict)


DynamicsConfig = Annotated[MockDynamicsConfig, Field(discriminator="type")]


# Generator Configs
class MockGeneratorConfig(BaseConfig):
    type: Literal["mock"] = "mock"
    n_candidates: int = 5
    params: dict[str, Any] = Field(default_factory=dict)


GeneratorConfig = Annotated[MockGeneratorConfig, Field(discriminator="type")]


# Validator Configs
class MockValidatorConfig(BaseConfig):
    type: Literal["mock"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)


ValidatorConfig = Annotated[MockValidatorConfig, Field(discriminator="type")]


# Selector Configs
class MockSelectorConfig(BaseConfig):
    type: Literal["mock"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)


SelectorConfig = Annotated[MockSelectorConfig, Field(discriminator="type")]


# Global Config
class GlobalConfig(BaseConfig):
    workdir: Path
    max_cycles: int = 10
    initial_structure_path: Path | None = None

    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    generator: GeneratorConfig
    validator: ValidatorConfig
    selector: SelectorConfig

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, v: Path) -> Path:
        # We don't necessarily create it here, but we check if it's a valid path structure
        # The Orchestrator will likely create it.
        # But for security, we might want to ensure it's not root or something dangerous if needed.
        # For now, just ensuring it is a Path is handled by Pydantic.
        return v

    @field_validator("max_cycles")
    @classmethod
    def validate_max_cycles(cls, v: int) -> int:
        if v <= 0:
            msg = "max_cycles must be positive"
            raise ValueError(msg)
        return v
