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
        if ".." in str(v):
            msg = "Path traversal ('..') is not allowed in work_dir."
            raise ValueError(msg)
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


class AdaptiveGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["adaptive"] = "adaptive"
    n_candidates: int = Field(default=20, ge=1)
    md_mc_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature_max: float = Field(default=1000.0, ge=0.0)


GeneratorConfig = Annotated[
    MockGeneratorConfig | RandomGeneratorConfig | AdaptiveGeneratorConfig,
    Field(discriminator="type"),
]


# --- Oracle Configs ---
class MockOracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"
    noise_std: float = Field(default=0.01, ge=0.0)


class DFTOracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["dft"] = "dft"
    code: Literal["qe", "vasp"] = "qe"
    kspacing: float = Field(default=0.04, ge=0.001)
    scf_accuracy: float = Field(default=1e-6, ge=1e-10)


OracleConfig = Annotated[
    MockOracleConfig | DFTOracleConfig,
    Field(discriminator="type"),
]


# --- Trainer Configs ---
class MockTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"
    potential_format: str = Field(default="yace")


class PacemakerTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["pacemaker"] = "pacemaker"
    potential_format: str = Field(default="yace")
    max_epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=100, ge=1)


TrainerConfig = Annotated[
    MockTrainerConfig | PacemakerTrainerConfig,
    Field(discriminator="type"),
]


# --- Dynamics Configs ---
class MockDynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"
    steps: int = Field(default=100, ge=1)


class LAMMPSDynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["lammps"] = "lammps"
    steps: int = Field(default=10000, ge=1)
    timestep: float = Field(default=0.001, ge=0.0)


class EONDynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["eon"] = "eon"
    temperature: float = Field(default=300.0, ge=0.0)


DynamicsConfig = Annotated[
    MockDynamicsConfig | LAMMPSDynamicsConfig | EONDynamicsConfig,
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
