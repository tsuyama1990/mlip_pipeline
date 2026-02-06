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

    # Espresso-specific fields (optional unless type="espresso")
    command: str | None = None
    pseudo_dir: Path | None = None
    pseudopotentials: dict[str, str] | None = None
    kspacing: float = Field(default=0.04, gt=0)
    # Default QE parameters. Users can override these via scf_params or config file.
    # Structure follows ASE Espresso input_data format (sections).
    @staticmethod
    def _default_qe_input_data() -> dict[str, dict[str, Any]]:
        return {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "tprnfor": True,
                "tstress": True,
                "disk_io": "none",
            },
            "system": {
                "ecutwfc": 40,
                "ecutrho": 160,
            },
            "electrons": {
                "mixing_beta": 0.7,
                "conv_thr": 1.0e-6,
            },
        }

    default_input_data: dict[str, dict[str, Any]] = Field(default_factory=_default_qe_input_data)
    # User overrides for specific parameters (can be flat or nested, adapter handles merging)
    scf_params: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=10, ge=1)

    @field_validator("command")
    @classmethod
    def check_command_security(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Basic check for shell injection characters
        forbidden_chars = [";", "|", "&", "`", "$(", ">", "<"]
        if any(char in v for char in forbidden_chars):
            msg = f"Command contains forbidden characters: {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def check_espresso_config(self) -> "OracleConfig":
        if self.type == "espresso":
            if not self.command:
                msg = "Field 'command' is required for espresso oracle"
                raise ValueError(msg)
            if self.pseudo_dir is None:
                msg = "Field 'pseudo_dir' is required for espresso oracle"
                raise ValueError(msg)
            if self.pseudopotentials is None:
                msg = "Field 'pseudopotentials' is required for espresso oracle"
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
