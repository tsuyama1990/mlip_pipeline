from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "mock"


class GeneratorConfig(ComponentConfig):
    name: Literal["adaptive", "random", "mock"] = "mock"
    n_structures: int = Field(default=10, gt=0)

    # Mock specific parameters - REQUIRED for reproducibility and explicit config
    cell_size: float = Field(..., gt=0)
    n_atoms: int = Field(..., gt=0)
    atomic_numbers: list[int] = Field(..., min_length=1)

    # Spec parameters
    md_mc_ratio: float | None = None
    temperature_schedule: dict[str, Any] | None = None
    defect_density: float | None = None
    strain_range: float | None = None


class OracleConfig(ComponentConfig):
    name: Literal["qe", "vasp", "mock"] = "mock"

    # Spec parameters
    kspacing: float | None = None
    mixing_beta: float | None = None
    smearing: float | None = None
    pseudopotentials: dict[str, str] | None = None


class TrainerConfig(ComponentConfig):
    name: Literal["pacemaker", "mock"] = "mock"
    max_num_epochs: int = 50
    energy_rmse_threshold: float | None = None
    force_rmse_threshold: float | None = None


class DynamicsConfig(ComponentConfig):
    name: Literal["lammps", "mock"] = "mock"
    uncertainty_threshold: float = 5.0

    # Mock specific - REQUIRED
    selection_rate: float = Field(..., ge=0.0, le=1.0)


class ValidatorConfig(ComponentConfig):
    name: Literal["standard", "mock"] = "mock"
    test_set_ratio: float = 0.1


class ComponentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator: GeneratorConfig
    oracle: OracleConfig
    trainer: TrainerConfig
    dynamics: DynamicsConfig
    validator: ValidatorConfig


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workdir: Path
    max_cycles: int = Field(gt=0)
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    components: ComponentsConfig
