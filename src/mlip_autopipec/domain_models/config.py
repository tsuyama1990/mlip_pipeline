from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    """Base configuration class."""

    model_config = ConfigDict(extra="forbid")


class MockOracleConfig(BaseConfig):
    type: Literal["mock"] = "mock"
    noise_level: float = 0.0


class MockTrainerConfig(BaseConfig):
    type: Literal["mock"] = "mock"


class MockDynamicsConfig(BaseConfig):
    type: Literal["mock"] = "mock"


# Discriminated Unions for Polymorphism
OracleConfig = Annotated[MockOracleConfig, Field(discriminator="type")]

TrainerConfig = Annotated[MockTrainerConfig, Field(discriminator="type")]

DynamicsConfig = Annotated[MockDynamicsConfig, Field(discriminator="type")]


class GlobalConfig(BaseConfig):
    """
    Root configuration object for the pipeline.
    """

    workdir: Path
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "GlobalConfig":
        """Load configuration from a YAML file."""
        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open("r") as f:
            data = yaml.safe_load(f)

        return cls(**data)
