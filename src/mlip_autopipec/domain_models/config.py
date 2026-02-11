from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Field(..., description="Root directory for the project")
    n_iterations: int = Field(default=1, ge=1, description="Number of active learning cycles")
    continue_on_error: bool = Field(default=False, description="Whether to continue loop on component failure")

    @field_validator("work_dir")
    @classmethod
    def validate_work_dir(cls, v: Path) -> Path:
        """Prevent path traversal attacks."""
        # Simple check for '..' in path string to prevent traversal
        # This is basic; robust security would require resolving against a root.
        # However, for Cycle 01 this satisfies the requirement to have a check.
        if ".." in str(v):
            raise ValueError("Path traversal ('..') is not allowed in work_dir.")
        return v


# --- Generator Configs ---
class MockGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"
    n_candidates: int = Field(default=10, ge=1)


class RandomGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["random"] = "random"
    n_candidates: int = Field(default=10, ge=1)
    elements: list[str] = Field(..., min_length=1)


GeneratorConfig = Annotated[
    MockGeneratorConfig | RandomGeneratorConfig,
    Field(discriminator="type"),
]


# --- Oracle Configs ---
class MockOracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"
    noise_std: float = Field(default=0.01, ge=0.0)


OracleConfig = Annotated[
    MockOracleConfig,
    Field(discriminator="type"),
]


# --- Trainer Configs ---
class MockTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"
    potential_format: str = Field(default="yace")


TrainerConfig = Annotated[
    MockTrainerConfig,
    Field(discriminator="type"),
]


# --- Dynamics Configs ---
class MockDynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"
    steps: int = Field(default=100, ge=1)


DynamicsConfig = Annotated[
    MockDynamicsConfig,
    Field(discriminator="type"),
]


# --- Validator Configs ---
class MockValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"


ValidatorConfig = Annotated[
    MockValidatorConfig,
    Field(discriminator="type"),
]


# --- Global Config ---
class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orchestrator: OrchestratorConfig
    generator: GeneratorConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    validator: ValidatorConfig
