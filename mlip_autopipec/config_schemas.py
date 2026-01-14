# ruff: noqa: D101
"""Configuration schemas for MLIP-AutoPipe.

Defines the data structures for user input and internal system configuration.
"""

from enum import Enum
from typing import Literal

from ase import Atoms
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

# =============================================================================
# User-Facing Configuration Schemas
# =============================================================================


class SimulationGoal(str, Enum):
    """Enumeration of supported simulation goals."""

    MELT_QUENCH = "melt_quench"
    ELASTIC = "elastic"
    VACANCY_DIFFUSION = "vacancy_diffusion"


class TargetSystem(BaseModel):
    """Defines the chemical system to be simulated."""

    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(
        ...,
        min_length=1,
        description="A list of chemical symbols of the constituent elements.",
    )
    composition: dict[str, float] = Field(
        ...,
        description=(
            "A dictionary mapping element symbols to their atomic fractions. "
            "The sum of fractions must be 1.0."
        ),
    )

    @field_validator("composition")
    @classmethod
    def validate_composition_fractions(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that the composition fractions sum to 1.0."""
        if not abs(sum(v.values()) - 1.0) < 1e-6:
            raise ValueError("Composition fractions must sum to 1.0")
        return v

    @field_validator("composition")
    @classmethod
    def validate_composition_elements(
        cls, v: dict[str, float], info: ValidationInfo
    ) -> dict[str, float]:
        """Validate that the elements in composition match the elements list."""
        if "elements" in info.data and set(v.keys()) != set(info.data["elements"]):
            raise ValueError("Elements in composition must match the elements list")
        return v


class Resources(BaseModel):
    """Defines the computational resources for the simulation."""

    model_config = ConfigDict(extra="forbid")

    dft_cores: int = Field(
        1, ge=1, description="Number of CPU cores for each DFT calculation."
    )
    md_cores: int = Field(
        1, ge=1, description="Number of CPU cores for each MD simulation."
    )
    max_dft_calculations: int = Field(
        100,
        ge=1,
        description="Maximum number of DFT calculations to perform in a run.",
    )


class UserConfig(BaseModel):
    """User-facing configuration for an MLIP-AutoPipe workflow."""

    model_config = ConfigDict(extra="forbid")

    target_system: TargetSystem = Field(
        ..., description="The chemical system to be simulated."
    )
    simulation_goal: SimulationGoal = Field(
        ..., description="The primary scientific goal of the simulation campaign."
    )
    resources: Resources = Field(
        default_factory=Resources,
        description="Computational resources for the workflow.",
    )


# =============================================================================
# System-Internal Configuration Schemas
# =============================================================================


class DFTControl(BaseModel):
    """Parameters for the &CONTROL namelist in Quantum Espresso."""

    model_config = ConfigDict(extra="forbid")
    calculation: str = Field("scf", description="Type of calculation.")


class DFTSystem(BaseModel):
    """Parameters for the &SYSTEM namelist in Quantum Espresso."""

    model_config = ConfigDict(extra="forbid")
    nat: int | None = Field(None, ge=1)
    ntyp: int | None = Field(None, ge=1)
    ecutwfc: float = Field(60.0)
    nspin: int = Field(1)
    occupations: str = Field("smearing")
    smearing: str = Field("mv")
    degauss: float = Field(0.01)


class DFTElectrons(BaseModel):
    """Parameters for the &ELECTRONS namelist in Quantum Espresso."""

    model_config = ConfigDict(extra="forbid")
    mixing_beta: float = Field(0.7)
    conv_thr: float = Field(1.0e-10)


class DFTInput(BaseModel):
    """Comprehensive container for Quantum Espresso input parameters."""

    model_config = ConfigDict(extra="forbid")
    pseudopotentials: dict[str, str] = Field(
        ..., description="Mapping from element symbol to pseudopotential filename."
    )
    control: DFTControl = Field(default_factory=DFTControl)
    system: DFTSystem = Field(default_factory=DFTSystem)
    electrons: DFTElectrons = Field(default_factory=DFTElectrons)

    @field_validator("pseudopotentials")
    @classmethod
    def validate_elements(cls, values: dict[str, str]) -> dict[str, str]:
        """Validate that the keys are valid chemical symbols."""
        from ase.data import chemical_symbols

        for key in values:
            if key not in chemical_symbols:
                raise ValueError(f"'{key}' is not a valid chemical symbol.")
        return values


class DFTExecutable(BaseModel):
    """Defines the command and working directory for the DFT executable."""

    model_config = ConfigDict(extra="forbid")
    command: str = Field("pw.x", description="The Quantum Espresso executable to run.")


class DFTConfig(BaseModel):
    """All parameters related to the DFT engine."""

    model_config = ConfigDict(extra="forbid")
    executable: DFTExecutable = Field(default_factory=DFTExecutable)
    input: DFTInput


class AlloyParams(BaseModel):
    """Parameters for generating alloy structures."""

    model_config = ConfigDict(extra="forbid")
    sqs_supercell_size: list[int] = Field(
        [2, 2, 2],
        description="Dimensions of the supercell for SQS generation.",
        min_length=3,
        max_length=3,
    )
    strain_magnitudes: list[float] = Field(
        [0.95, 1.0, 1.05],
        description="List of volumetric strain magnitudes to apply.",
    )
    rattle_std_devs: list[float] = Field(
        [0.0, 0.1, 0.2],
        description="Standard deviations for atomic position rattling.",
    )


class CrystalParams(BaseModel):
    """Parameters for generating crystal structures with defects."""

    model_config = ConfigDict(extra="forbid")
    defect_types: list[Literal["vacancy", "interstitial"]] = Field(
        ["vacancy"],
        description="Types of point defects to introduce.",
    )


class GeneratorParams(BaseModel):
    """Parameters for the Physics-Informed Generator."""

    model_config = ConfigDict(extra="forbid")
    alloy: AlloyParams = Field(default_factory=AlloyParams)
    crystal: CrystalParams = Field(default_factory=CrystalParams)


class SystemConfig(BaseModel):
    """The root internal configuration object for an MLIP-AutoPipe workflow."""

    model_config = ConfigDict(extra="forbid")
    target_system: TargetSystem
    dft: DFTConfig
    generator: GeneratorParams = Field(default_factory=GeneratorParams)
    db_path: str = Field(
        "mlip_database.db", description="Path to the central ASE database."
    )


class CalculationMetadata(BaseModel):
    """Schema for metadata to be stored with each calculation in the database."""

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(
        ..., description="The name of the workflow stage that produced the structure."
    )
    uuid: str = Field(..., description="A unique identifier for the calculation run.")


class AtomicStructure(BaseModel):
    """A wrapper for an ASE Atoms object to be used within the system."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    atoms: Atoms
