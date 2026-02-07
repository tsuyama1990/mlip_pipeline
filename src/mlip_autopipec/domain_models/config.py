from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


# --- Generator Configs ---
class MockGeneratorConfig(BaseModel):
    type: Literal["mock"] = "mock"
    initial_count: int = 10


GeneratorConfig = Annotated[MockGeneratorConfig, Field(discriminator="type")]


# --- Oracle Configs ---
class MockOracleConfig(BaseModel):
    type: Literal["mock"] = "mock"
    noise_level: float = 0.01


class QuantumEspressoConfig(BaseModel):
    type: Literal["qe"] = "qe"
    command: str
    pseudo_dir: Path
    pseudopotentials: dict[str, str]
    kspacing: float = 0.04
    smearing_width: float = 0.02


OracleConfig = Annotated[MockOracleConfig | QuantumEspressoConfig, Field(discriminator="type")]


# --- Trainer Configs ---
class MockTrainerConfig(BaseModel):
    type: Literal["mock"] = "mock"


TrainerConfig = Annotated[MockTrainerConfig, Field(discriminator="type")]


# --- Dynamics Configs ---
class MockDynamicsConfig(BaseModel):
    type: Literal["mock"] = "mock"


DynamicsConfig = Annotated[MockDynamicsConfig, Field(discriminator="type")]


# --- Validator Configs ---
class MockValidatorConfig(BaseModel):
    type: Literal["mock"] = "mock"


ValidatorConfig = Annotated[MockValidatorConfig, Field(discriminator="type")]


# --- Selector Configs ---
class MockSelectorConfig(BaseModel):
    type: Literal["mock"] = "mock"


SelectorConfig = Annotated[MockSelectorConfig, Field(discriminator="type")]


# --- Global Config ---
class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir: Path
    max_cycles: int = 10
    initial_structure_path: Path | None = None

    generator: GeneratorConfig = Field(default_factory=MockGeneratorConfig)
    oracle: OracleConfig = Field(default_factory=MockOracleConfig)
    trainer: TrainerConfig = Field(default_factory=MockTrainerConfig)
    dynamics: DynamicsConfig = Field(default_factory=MockDynamicsConfig)
    validator: ValidatorConfig = Field(default_factory=MockValidatorConfig)
    selector: SelectorConfig = Field(default_factory=MockSelectorConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "GlobalConfig":
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
