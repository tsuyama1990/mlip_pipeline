from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mlip_autopipec.constants import YACE_EXTENSION


class ExplorerConfig(BaseModel):
    """
    Configuration for the Explorer component.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "random", "md"] = "mock"
    n_structures: int = Field(default=2, ge=1, description="Number of candidate structures to generate per cycle.")


class OracleConfig(BaseModel):
    """
    Configuration for the Oracle component.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "espresso"] = "mock"

    # Espresso specific configuration
    command: str | None = Field(default=None, description="Command to run the Oracle (e.g., 'pw.x').")
    pseudo_dir: Path | None = Field(default=None, description="Directory containing pseudopotentials.")
    pseudopotentials: dict[str, str] | None = Field(default=None, description="Mapping of element symbol to pseudopotential filename.")
    kspacing: float = Field(default=0.04, description="K-point spacing in inverse Angstroms.")
    scf_params: dict[str, Any] = Field(default_factory=dict, description="Additional SCF parameters for the calculator.")
    recovery_recipes: list[dict[str, Any]] = Field(default_factory=list, description="List of recovery strategies for SCF convergence failures.")

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
    """
    Configuration for the Trainer component.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock", "pacemaker"] = "mock"
    potential_output_name: str = Field(default=f"potential{YACE_EXTENSION}", description="Filename for the output potential.")

    @field_validator("potential_output_name")
    @classmethod
    def check_extension(cls, v: str) -> str:
        if not v.endswith(YACE_EXTENSION):
            msg = f"Potential output name must end with {YACE_EXTENSION}"
            raise ValueError(msg)
        return v


class ValidatorConfig(BaseModel):
    """
    Configuration for the Validator component.
    """
    model_config = ConfigDict(extra="forbid")
    type: Literal["mock"] = "mock"


class GlobalConfig(BaseModel):
    """
    Global configuration for the MLIP pipeline.
    """
    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Field(..., description="Working directory for the pipeline.")
    max_cycles: int = Field(..., ge=1, description="Maximum number of active learning cycles.")
    random_seed: int = Field(..., description="Random seed for reproducibility.")
    max_accumulated_structures: int = Field(default=10000, ge=0, description="Maximum number of structures to accumulate.")

    initial_potential: Path | None = Field(default=None, description="Path to an initial potential file.")

    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
