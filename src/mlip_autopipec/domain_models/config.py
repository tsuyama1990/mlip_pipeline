import os
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.constants import (
    DEFAULT_BAROSTAT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CUTOFF,
    DEFAULT_DYNAMICS_STEPS,
    DEFAULT_ELEMENTS,
    DEFAULT_ENERGY_WEIGHT,
    DEFAULT_FORCES_WEIGHT,
    DEFAULT_K_POINTS_DENSITY,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MD_MC_RATIO,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PROJECT_NAME,
    DEFAULT_SCF_K_POINTS,
    DEFAULT_SEED,
    DEFAULT_SMEARING,
    DEFAULT_STRESS_WEIGHT,
    DEFAULT_THERMOSTAT,
    DEFAULT_TIMESTEP,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_VALID_FRACTION,
)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = cast(
        Literal["DEBUG", "INFO", "WARNING", "ERROR"], DEFAULT_LOG_LEVEL
    )
    file_path: Path = Path(os.getenv("MLIP_LOG_FILENAME", "mlip_pipeline.log"))


class PotentialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(default_factory=lambda: DEFAULT_ELEMENTS)
    cutoff: float = Field(default=DEFAULT_CUTOFF)
    seed: int = Field(default=DEFAULT_SEED)

    @field_validator("cutoff")
    @classmethod
    def validate_cutoff(cls, v: float) -> float:
        if v <= 0:
            msg = "Cutoff must be greater than 0"
            raise ValueError(msg)
        return v

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "Elements list cannot be empty"
            raise ValueError(msg)
        return v


class AdaptivePolicyConfig(BaseModel):
    """Configuration for Adaptive Exploration Policy."""
    model_config = ConfigDict(extra="forbid")

    md_mc_ratio: float = Field(default=DEFAULT_MD_MC_RATIO, description="Ratio of MC steps to MD steps")
    temperature_schedule: list[float] = Field(default_factory=lambda: [300.0, 600.0, 900.0])
    pressure_schedule: list[float] = Field(default_factory=lambda: [0.0])

    @field_validator("md_mc_ratio")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        if not (0 <= v <= 1):
            msg = "MD/MC ratio must be between 0 and 1"
            raise ValueError(msg)
        return v


class ExplorationConfig(BaseModel):
    """Configuration for Structure Exploration/Generation."""
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["random", "template", "adaptive"] = "random"
    supercell_size: list[int] = Field(default_factory=lambda: [1, 1, 1])
    rattle_amplitude: float = Field(default=0.1, ge=0.0)
    num_candidates: int = Field(default=1, ge=1)
    composition: str | None = Field(default=None, description="Chemical formula for cold start (e.g. 'Si')")
    policy: AdaptivePolicyConfig = Field(default_factory=AdaptivePolicyConfig)

    @field_validator("supercell_size")
    @classmethod
    def validate_supercell(cls, v: list[int]) -> list[int]:
        if len(v) != 3 or any(i < 1 for i in v):
            msg = "Supercell size must be 3 positive integers"
            raise ValueError(msg)
        return v


class OracleConfig(BaseModel):
    """Configuration for DFT Oracle (Quantum Espresso / VASP)."""
    model_config = ConfigDict(extra="forbid")

    code: Literal["qe", "vasp", "fake"] = "qe"
    k_points_density: float | None = Field(default=DEFAULT_K_POINTS_DENSITY, description="K-points density in 1/A")
    scf_k_points: list[int] = Field(default_factory=lambda: DEFAULT_SCF_K_POINTS, description="Explicit k-points grid")
    smearing: float = Field(default=DEFAULT_SMEARING, description="Smearing width in Ry")
    pseudopotentials: dict[str, str] = Field(default_factory=dict, description="Path to pseudopotentials per element")
    command: str = Field(default="pw.x", description="Command to run the code")


class TrainerConfig(BaseModel):
    """Configuration for Potential Training (MACE/Pacemaker)."""
    model_config = ConfigDict(extra="forbid")

    engine: Literal["mace", "pacemaker", "fake"] = "mace"
    max_epochs: int = Field(default=DEFAULT_TRAIN_EPOCHS)
    model_size: str = Field(default=DEFAULT_MODEL_SIZE)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE)
    valid_fraction: float = Field(default=DEFAULT_VALID_FRACTION)
    energy_weight: float = Field(default=DEFAULT_ENERGY_WEIGHT)
    forces_weight: float = Field(default=DEFAULT_FORCES_WEIGHT)
    stress_weight: float = Field(default=DEFAULT_STRESS_WEIGHT)

    @field_validator("max_epochs")
    @classmethod
    def validate_epochs(cls, v: int) -> int:
        if v <= 0:
            msg = "Max epochs must be positive"
            raise ValueError(msg)
        return v


class DynamicsConfig(BaseModel):
    """Configuration for MD/kMC Dynamics."""
    model_config = ConfigDict(extra="forbid")
    timestep: float = Field(default=DEFAULT_TIMESTEP)
    n_steps: int = Field(default=DEFAULT_DYNAMICS_STEPS)
    thermostat: str = Field(default=DEFAULT_THERMOSTAT)
    barostat: str = Field(default=DEFAULT_BAROSTAT)


class OrchestratorConfig(BaseModel):
    """Configuration for Orchestrator."""
    model_config = ConfigDict(extra="forbid")
    max_cycles: int = 10
    state_file: Path = Path(os.getenv("MLIP_STATE_FILENAME", "workflow_state.json"))
    active_learning_strategy: Literal["uncertainty", "random"] = "uncertainty"


class Config(BaseModel):
    """Root configuration object."""
    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(default=DEFAULT_PROJECT_NAME)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    potential: PotentialConfig = Field(default_factory=PotentialConfig)
    structure_gen: ExplorationConfig = Field(default_factory=ExplorationConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        if not v.strip():
            msg = "Project name cannot be empty"
            raise ValueError(msg)
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        from mlip_autopipec.infrastructure import io
        data = io.load_yaml(path)
        return cls(**data)

    @classmethod
    def get_default_template(cls) -> dict[str, Any]:
        """Return a default configuration template."""
        return {
            "project_name": DEFAULT_PROJECT_NAME,
            "potential": {
                "elements": DEFAULT_ELEMENTS,
                "cutoff": DEFAULT_CUTOFF,
                "seed": DEFAULT_SEED
            },
            "logging": {
                "level": DEFAULT_LOG_LEVEL,
            },
            "trainer": {
                "max_epochs": DEFAULT_TRAIN_EPOCHS,
                "batch_size": DEFAULT_BATCH_SIZE
            }
        }
