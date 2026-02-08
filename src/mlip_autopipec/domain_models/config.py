from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "mock"


class BaseGeneratorConfig(ComponentConfig):
    n_structures: int = Field(default=10, gt=0)


class MockGeneratorConfig(BaseGeneratorConfig):
    name: Literal[GeneratorType.MOCK] = GeneratorType.MOCK
    cell_size: float = Field(..., gt=0)
    n_atoms: int = Field(..., gt=0)
    atomic_numbers: list[int] = Field(..., min_length=1)


class AdaptiveGeneratorConfig(BaseGeneratorConfig):
    name: Literal[GeneratorType.ADAPTIVE] = GeneratorType.ADAPTIVE
    element: str
    crystal_structure: str
    strain_range: float = 0.05
    rattle_strength: float = 0.01
    surface_indices: list[list[int]] = Field(
        default_factory=lambda: [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    )
    vacuum: float = 10.0
    supercell_dim: int = 2
    policy_ratios: dict[str, float] = Field(
        default_factory=lambda: {"cycle0_bulk": 0.6, "cycle0_surface": 0.4}
    )


# Union type for generator configuration
# NOTE: The variable name GeneratorConfig is overloaded here to serve as the union type alias
# for external usage, but specific configs should be used where concrete types are known.
GeneratorConfig = MockGeneratorConfig | AdaptiveGeneratorConfig


class OracleConfig(ComponentConfig):
    name: OracleType = OracleType.MOCK

    # Spec parameters
    kspacing: float | None = None
    mixing_beta: float | None = None
    smearing: float | None = None
    pseudopotentials: dict[str, str] | None = None


class TrainerConfig(ComponentConfig):
    name: TrainerType = TrainerType.MOCK
    max_num_epochs: int = 50
    energy_rmse_threshold: float | None = None
    force_rmse_threshold: float | None = None


class DynamicsConfig(ComponentConfig):
    name: DynamicsType = DynamicsType.MOCK
    uncertainty_threshold: float = 5.0

    # Mock specific - REQUIRED
    selection_rate: float = Field(..., ge=0.0, le=1.0)


class ValidatorConfig(ComponentConfig):
    name: ValidatorType = ValidatorType.MOCK
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

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, v: Path) -> Path:
        """Ensure workdir is a valid path and its parent exists."""
        try:
            # We don't necessarily want to create it here, but checking parent existence is a good sanity check
            # if we expect the user to provide a valid location.
            # Or at least check if it's absolute or we can resolve it.
            resolved = v.resolve()
            if not resolved.parent.exists():
                msg = f"Workdir parent directory does not exist: {resolved.parent}"
                raise ValueError(msg)
        except OSError as e:
            msg = f"Invalid workdir path: {v}"
            raise ValueError(msg) from e
        return v
