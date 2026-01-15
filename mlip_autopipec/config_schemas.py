# ruff: noqa: D101
"""Configuration schemas for MLIP-AutoPipe.

Defines the data structures for user input and internal system configuration.
"""

from enum import Enum
from typing import Literal

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
    # A dictionary is used here as it is the most appropriate data structure for
    # mapping dynamically determined element symbols to their pseudopotential
    # filenames. This approach is preferred over nested Pydantic models, which
    # would require a pre-defined and static set of fields (i.e., all possible
    # chemical symbols), making the schema inflexible.
    #
    # Data integrity is robustly maintained by the `validate_elements` validator
    # below, which programmatically ensures that every key in the dictionary is a
    # valid chemical symbol at runtime. This guarantees that the dictionary
    # remains well-formed and consistent with the expected data model without
    # sacrificing the necessary flexibility.
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

    @field_validator("pseudopotentials")
    @classmethod
    def validate_pseudopotential_paths(cls, values: dict[str, str]) -> dict[str, str]:
        """Validate the pseudopotential filenames to prevent path traversal."""
        import os

        for filename in values.values():
            if ".." in filename or os.path.isabs(filename):
                raise ValueError(
                    f"Pseudopotential '{filename}' must be a relative path and "
                    "cannot contain '..'"
                )
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
        default=[4, 4, 4],
        min_length=3,
        description="Supercell dimensions for SQS generation (e.g., [2, 2, 2]).",
    )
    strain_magnitudes: list[float] = Field(
        default=[0.95, 1.0, 1.05],
        description="List of isotropic strain magnitudes to apply.",
    )
    rattle_std_devs: list[float] = Field(
        default=[0.05, 0.1],
        description="List of standard deviations for atomic rattling.",
    )


class CrystalParams(BaseModel):
    """Parameters for generating crystal structures with defects."""

    model_config = ConfigDict(extra="forbid")

    defect_types: list[Literal["vacancy", "interstitial"]] = Field(
        default=["vacancy"], description="List of defect types to generate."
    )


class GeneratorParams(BaseModel):
    """Parameters for the Physics-Informed Generator (Module A)."""

    model_config = ConfigDict(extra="forbid")

    alloy_params: AlloyParams = Field(
        default_factory=AlloyParams, description="Parameters for alloy generation."
    )
    crystal_params: CrystalParams = Field(
        default_factory=CrystalParams,
        description="Parameters for crystal defect generation.",
    )


class SurrogateModelParams(BaseModel):
    """Parameters for the surrogate model screening."""

    model_config = ConfigDict(extra="forbid")
    model_path: str = Field(
        ..., description="Path to the pre-trained surrogate model file."
    )
    energy_threshold_ev: float = Field(
        -2.0,
        description=(
            "Energy per atom threshold in eV. Structures with higher energy will be "
            "discarded."
        ),
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate the model path to prevent path traversal."""
        import os

        if ".." in v or os.path.isabs(v):
            raise ValueError(
                "model_path must be a relative path and cannot contain '..'"
            )
        return v


class SOAPParams(BaseModel):
    """Hyperparameters for the SOAP descriptor."""

    model_config = ConfigDict(extra="forbid")
    n_max: int = Field(8, description="Number of radial basis functions.")
    l_max: int = Field(6, description="Maximum degree of spherical harmonics.")
    r_cut: float = Field(5.0, description="Cutoff radius for the local environment.")
    atomic_sigma: float = Field(
        0.5, description="Standard deviation of the Gaussian smearing."
    )


class FPSParams(BaseModel):
    """Parameters for the Farthest Point Sampling algorithm."""

    model_config = ConfigDict(extra="forbid")
    num_structures_to_select: int = Field(
        200, ge=1, description="The final number of structures to select."
    )
    soap_params: SOAPParams = Field(
        default_factory=SOAPParams, description="SOAP descriptor hyperparameters."
    )


class ExplorerParams(BaseModel):
    """Parameters for the Surrogate Explorer and Selector (Module B)."""

    model_config = ConfigDict(extra="forbid")
    surrogate_model: SurrogateModelParams
    fps: FPSParams = Field(default_factory=FPSParams, description="Parameters for FPS.")


class LossWeights(BaseModel):
    """Defines relative weights for energy, forces, and stress in the loss function."""

    model_config = ConfigDict(extra="forbid")
    energy: float = Field(1.0, gt=0)
    forces: float = Field(100.0, gt=0)
    stress: float = Field(10.0, gt=0)


class ACEParams(BaseModel):
    """Hyperparameters for the Atomic Cluster Expansion (ACE) potential."""

    model_config = ConfigDict(extra="forbid")
    radial_basis: str = Field("chebyshev", description="The type of radial basis.")
    correlation_order: int = Field(
        3, ge=2, description="The body order of the potential."
    )
    element_dependent_cutoffs: bool = Field(
        False, description="Whether to use different cutoffs for different elements."
    )


class TrainerParams(BaseModel):
    """Parameters for the Pacemaker Trainer (Module D)."""

    model_config = ConfigDict(extra="forbid")
    loss_weights: LossWeights = Field(
        default_factory=LossWeights,
        description="Weights for the training loss function.",
    )
    ace_params: ACEParams = Field(
        default_factory=ACEParams, description="Hyperparameters for the ACE potential."
    )


class SystemConfig(BaseModel):
    """The root internal configuration object for an MLIP-AutoPipe workflow."""

    model_config = ConfigDict(extra="forbid")
    dft: DFTConfig
    generator: GeneratorParams = Field(default_factory=GeneratorParams)
    explorer: ExplorerParams | None = Field(
        default=None, description="Parameters for the Surrogate Explorer."
    )
    trainer: TrainerParams = Field(
        default_factory=TrainerParams,
        description="Parameters for the Pacemaker Trainer.",
    )
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
