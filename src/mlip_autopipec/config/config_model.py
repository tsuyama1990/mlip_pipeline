from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExplorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "random", "md"] = "mock"
    n_structures: int = Field(default=2, ge=1)


class OracleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "espresso"] = "mock"

    # Espresso specific configuration
    command: str | None = None
    pseudo_dir: Path | None = None
    pseudopotentials: dict[str, str] | None = None
    kspacing: float = 0.04
    scf_params: dict[str, Any] = Field(default_factory=dict)
    recovery_recipes: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def validate_command_security(cls, v: str | None) -> str | None:
        if v is None:
            return v
        dangerous_patterns = [">", "<", "|", ";", "&"]
        if any(p in v for p in dangerous_patterns):
            msg = f"Command contains dangerous characters: {v}"
            raise ValueError(msg)
        return v

    @field_validator("kspacing")
    @classmethod
    def validate_kspacing(cls, v: float) -> float:
        if v <= 0:
            msg = "kspacing must be positive"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_espresso_config(self) -> "OracleConfig":
        if self.type == "espresso":
            if not self.command:
                msg = "command is required for espresso oracle"
                raise ValueError(msg)
            if not self.pseudo_dir:
                msg = "pseudo_dir is required for espresso oracle"
                raise ValueError(msg)
            if not self.pseudopotentials:
                msg = "pseudopotentials are required for espresso oracle"
                raise ValueError(msg)
        return self


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "pacemaker"] = "mock"
    # Configurable output path for potential
    potential_output_name: str = "potential.yace"

    @field_validator("potential_output_name")
    @classmethod
    def check_extension(cls, v: str) -> str:
        if not v.endswith(".yace"):
            msg = "Potential output name must end with .yace"
            raise ValueError(msg)
        return v


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path
    max_cycles: int = Field(ge=1)
    random_seed: int
    max_accumulated_structures: int = Field(default=10000, ge=0)

    # Optional initial potential path
    initial_potential: Path | None = None

    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
