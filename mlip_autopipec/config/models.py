# FIXME: The above comment is a temporary workaround for a ruff bug.
# It should be removed once the bug is fixed.
# For more information, see: https://github.com/astral-sh/ruff/issues/10515
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    RootModel,
    ValidationInfo,
    field_validator,
    model_validator,
)

# User-Facing Models


class TargetSystem(BaseModel):
    elements: list[str]
    composition: dict[str, float]
    crystal_structure: str

    @field_validator("elements")
    def validate_elements(cls, elements: list[str]) -> list[str]:
        from ase.data import chemical_symbols

        for symbol in elements:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")
        return elements

    @model_validator(mode="after")
    def check_composition_keys_and_sum(self):
        if set(self.elements) != set(self.composition.keys()):
            raise ValueError("Composition keys must match the elements list.")
        if not abs(sum(self.composition.values()) - 1.0) < 1e-6:
            raise ValueError("Composition fractions must sum to 1.0.")
        return self


class SimulationGoal(BaseModel):
    type: Literal["melt_quench", "elastic", "diffusion"]
    temperature_range: tuple[float, float] | None = None


class UserInputConfig(BaseModel):
    project_name: str
    target_system: TargetSystem
    simulation_goal: SimulationGoal
    model_config = ConfigDict(extra="forbid")


# Internal-Facing Models


class CutoffConfig(BaseModel):
    wavefunction: float = Field(..., description="Plane-wave cutoff energy in Rydberg.", gt=0)
    density: float = Field(..., description="Charge density cutoff energy in Rydberg.", gt=0)
    model_config = ConfigDict(extra="forbid")


class FingerprintConfig(BaseModel):
    type: Literal["soap"] = "soap"
    soap_rcut: float = Field(5.0, gt=0)
    soap_nmax: int = Field(8, gt=0)
    soap_lmax: int = Field(6, gt=0)
    species: list[str] = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")


class ExplorerConfig(BaseModel):
    surrogate_model_path: str
    max_force_threshold: float = Field(10.0, gt=0)
    fingerprint: FingerprintConfig
    model_config = ConfigDict(extra="forbid")


class SmearingConfig(BaseModel):
    smearing_type: Literal["gauss", "mv", "marzari-vanderbilt"] = Field(
        "mv", description="Type of smearing to be used for metals."
    )
    degauss: float = Field(0.02, description="Smearing width in Rydberg.", gt=0)
    model_config = ConfigDict(extra="forbid")


class StartingMagnetization(RootModel[dict[str, float]]):
    @field_validator("root")
    def validate_symbols(cls, mag_values: dict[str, str]) -> dict[str, str]:
        from ase.data import chemical_symbols

        for symbol in mag_values:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")
        return mag_values


class MagnetismConfig(BaseModel):
    nspin: Literal[2] = 2
    starting_magnetization: StartingMagnetization = Field(
        ..., description="Initial magnetic moment for each atomic species."
    )
    model_config = ConfigDict(extra="forbid")


class Pseudopotentials(RootModel[dict[str, str]]):
    @field_validator("root")
    def validate_symbols(cls, pseudo_values: dict[str, str]) -> dict[str, str]:
        from ase.data import chemical_symbols

        for symbol in pseudo_values:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")
        return pseudo_values


class DFTInputParameters(BaseModel):
    calculation_type: Literal["scf"] = "scf"
    pseudopotentials: Pseudopotentials
    cutoffs: CutoffConfig
    k_points: tuple[int, int, int]
    smearing: SmearingConfig | None = None
    magnetism: MagnetismConfig | None = None
    mixing_beta: float = Field(0.7, gt=0.0, le=1.0)
    diagonalization: Literal["david", "cg"] = "david"

    @field_validator("k_points")
    @classmethod
    def k_points_must_be_positive(cls, k_points: tuple[int, int, int]) -> tuple[int, int, int]:
        if not all(k > 0 for k in k_points):
            raise ValueError("All k-point dimensions must be positive integers.")
        return k_points

    model_config = ConfigDict(extra="forbid")


class DFTExecutable(BaseModel):
    """Defines the command and working directory for the DFT executable."""

    model_config = ConfigDict(extra="forbid")
    command: str = Field("pw.x", description="The Quantum Espresso executable to run.")


class DFTRetryStrategy(BaseModel):
    """Defines the strategy for retrying failed DFT calculations."""

    model_config = ConfigDict(extra="forbid")
    max_retries: int = Field(
        3, ge=0, description="Maximum number of times to retry a failed calculation."
    )
    parameter_adjustments: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "A list of parameter adjustments to apply on each retry attempt. "
            "Each dict key is a dot-separated path to the parameter to change."
        ),
    )


class DFTInput(BaseModel):
    """Comprehensive container for Quantum Espresso input parameters."""
    # Add fields as needed to match usage in conftest.py
    pseudopotentials: dict[str, str]
    system: Any = None
    # NOTE: This model seems to be a remnant of config_schemas.py structure.
    # config/models.py uses DFTInputParameters.
    # I am adding it here to satisfy tests/conftest.py which uses it.
    model_config = ConfigDict(extra="allow") # Allowing extra for now to be safe with unknown fields


class DFTSystem(BaseModel):
    """Parameters for the &SYSTEM namelist in Quantum Espresso."""
    model_config = ConfigDict(extra="allow")
    nat: int | None = Field(None, ge=1)
    ntyp: int | None = Field(None, ge=1)
    ecutwfc: float = Field(60.0)
    nspin: int = Field(1)

class DFTConfig(BaseModel):
    """Configuration for the DFT Factory."""
    # Modified to support both new and legacy fields temporarily or merge them.
    # The new code uses dft_input_params.
    # The old code (conftest) uses executable, input, retry_strategy.

    # New field
    dft_input_params: DFTInputParameters | None = None

    # Legacy fields (imported from config_schemas)
    executable: DFTExecutable = Field(default_factory=DFTExecutable)
    input: DFTInput | None = None
    retry_strategy: DFTRetryStrategy = Field(default_factory=DFTRetryStrategy)

    model_config = ConfigDict(extra="ignore")


class MDConfig(BaseModel):
    ensemble: Literal["nvt", "npt"] = "nvt"
    temperature: float = Field(300.0, gt=0)
    timestep: float = Field(1.0, gt=0)
    run_duration: int = Field(1000, gt=0)
    model_config = ConfigDict(extra="forbid")


class UncertaintyConfig(BaseModel):
    threshold: float = Field(5.0, gt=0)
    embedding_cutoff: float = Field(8.0, gt=0)
    masking_cutoff: float = Field(5.0, gt=0)
    model_config = ConfigDict(extra="forbid")

    @field_validator("masking_cutoff")
    @classmethod
    def masking_must_be_less_than_embedding(cls, masking_cutoff: float, info: ValidationInfo) -> float:
        if "embedding_cutoff" in info.data and masking_cutoff >= info.data["embedding_cutoff"]:
            raise ValueError("masking_cutoff must be smaller than embedding_cutoff.")
        return masking_cutoff




class InferenceConfig(BaseModel):
    lammps_executable: FilePath | None = None
    potential_path: FilePath | None = None
    md_params: MDConfig = Field(default_factory=MDConfig)
    uncertainty_params: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    model_config = ConfigDict(extra="forbid")


class LossWeights(BaseModel):
    energy: float = Field(1.0, gt=0)
    forces: float = Field(100.0, gt=0)
    stress: float = Field(10.0, gt=0)
    model_config = ConfigDict(extra="forbid")


class ACEParams(BaseModel):
    """Hyperparameters for the Atomic Cluster Expansion (ACE) potential."""

    model_config = ConfigDict(extra="forbid")
    radial_basis: str = Field("chebyshev", description="The type of radial basis.")
    correlation_order: int = Field(3, ge=2, description="The body order of the potential.")
    element_dependent_cutoffs: bool = Field(
        False, description="Whether to use different cutoffs for different elements."
    )


class TrainingConfig(BaseModel):
    pacemaker_executable: FilePath | None = None
    data_source_db: Path
    template_file: FilePath | None = None
    delta_learning: bool = True
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    ace_params: ACEParams = Field(
        default_factory=lambda: ACEParams(
            radial_basis="radial", correlation_order=3, element_dependent_cutoffs=False
        )
    )
    model_config = ConfigDict(extra="forbid")


class WorkflowConfig(BaseModel):
    checkpoint_filename: str = "checkpoint.json"
    model_config = ConfigDict(extra="forbid")


class SurrogateModelParams(BaseModel):
    """Parameters for the surrogate model screening."""

    model_config = ConfigDict(extra="forbid")
    model_path: str = Field(..., description="Path to the pre-trained surrogate model file.")
    energy_threshold_ev: float = Field(
        -2.0,
        description=(
            "Energy per atom threshold in eV. Structures with higher energy will be discarded."
        ),
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate the model path to prevent path traversal."""
        import os

        if ".." in v or os.path.isabs(v):
            raise ValueError("model_path must be a relative path and cannot contain '..'")
        return v

class SOAPParams(BaseModel):
    """Hyperparameters for the SOAP descriptor."""

    model_config = ConfigDict(extra="forbid")
    n_max: int = Field(8, description="Number of radial basis functions.")
    l_max: int = Field(6, description="Maximum degree of spherical harmonics.")
    r_cut: float = Field(5.0, description="Cutoff radius for the local environment.")
    atomic_sigma: float = Field(0.5, description="Standard deviation of the Gaussian smearing.")

class FPSParams(BaseModel):
    """Parameters for the Farthest Point Sampling algorithm."""

    model_config = ConfigDict(extra="forbid")
    num_structures_to_select: int = Field(
        200, ge=1, description="The final number of structures to select."
    )
    soap_params: SOAPParams = Field(
        default_factory=SOAPParams, description="SOAP descriptor hyperparameters."
    )

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

class DaskConfig(BaseModel):
    """Configuration for the Dask distributed task scheduler."""

    model_config = ConfigDict(extra="forbid")
    scheduler_address: str | None = Field(
        None,
        description=(
            "The address of the Dask scheduler (e.g., 'tcp://127.0.0.1:8786'). "
            "If None, a local cluster will be used."
        ),
    )

class ExplorerParams(BaseModel):
    """Parameters for the Surrogate Explorer and Selector (Module B)."""

    model_config = ConfigDict(extra="forbid")
    surrogate_model: SurrogateModelParams
    fps: FPSParams = Field(default_factory=FPSParams, description="Parameters for FPS.")

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

    project_name: str | None = None # Made optional to fit legacy tests if needed
    run_uuid: UUID | None = None

    # Legacy fields
    target_system: Any = None
    dft: DFTConfig | None = None
    db_path: str = "mlip_database.db"

    # New fields
    workflow_config: WorkflowConfig = Field(default_factory=WorkflowConfig)
    dft_config: DFTConfig | None = None # Renamed or aliased?
    # Wait, dft_config in new model was type DFTConfig.
    # In old model 'dft' was type DFTConfig.
    # I should align them.

    explorer_config: ExplorerConfig | None = None
    training_config: TrainingConfig | None = None
    inference_config: InferenceConfig | None = None

    # Generator params from schemas
    generator: GeneratorParams | None = None
    explorer: ExplorerParams | None = None
    trainer: TrainerParams | None = None
    inference: InferenceConfig | None = None
    dask: DaskConfig | None = None

    model_config = ConfigDict(extra="ignore")


# Data Transfer Objects


class UncertaintyMetadata(BaseModel):
    uncertain_timestep: int
    uncertain_atom_id: int
    uncertain_atom_index_in_original_cell: int
    model_config = ConfigDict(extra="forbid")


class UncertainStructure(BaseModel):
    """
    Data transfer object for a structure with high uncertainty.

    Attributes:
        atoms: The atomic structure flagged for high uncertainty. `object` is
               used as the type hint because `ase.Atoms` is not directly
               serializable. Type safety is enforced by a Pydantic validator.
        force_mask: A NumPy array indicating which atoms' forces should be
                    included in training. `object` is used as the type hint
                    for compatibility, with runtime validation.
        metadata: Additional metadata about the uncertainty event.
    """

    atoms: object
    force_mask: object
    metadata: UncertaintyMetadata
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("atoms")
    @classmethod
    def validate_atoms_type(cls, atoms_obj: Any) -> Any:
        try:
            from ase import Atoms
        except ImportError as e:
            raise ImportError("ASE is required for this model.") from e
        if not isinstance(atoms_obj, Atoms):
            raise TypeError("Field 'atoms' must be an instance of ase.Atoms.")
        return atoms_obj

    @field_validator("force_mask")
    @classmethod
    def validate_force_mask(cls, force_mask_obj: Any, info: ValidationInfo) -> Any:
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("NumPy is required for this model.") from e
        if not isinstance(force_mask_obj, np.ndarray):
            raise TypeError("Field 'force_mask' must be a NumPy array.")
        if "atoms" in info.data and len(force_mask_obj) != len(info.data["atoms"]):
            raise ValueError("force_mask must have the same length as the number of atoms.")
        return force_mask_obj


class DFTJob(BaseModel):
    """
    Represents a single, self-contained DFT job to be executed.

    Attributes:
        atoms: The atomic structure to be calculated. `object` is used as the
               type hint because `ase.Atoms` is not directly serializable.
               Type safety is enforced by a Pydantic validator.
        params: The validated input parameters for the DFT calculation.
        job_id: A unique identifier for this specific job.
    """

    atoms: object
    params: DFTInputParameters
    job_id: UUID = Field(default_factory=uuid4)

    @field_validator("atoms")
    def validate_atoms_type(cls, atoms_obj: Any) -> Any:
        try:
            from ase import Atoms
        except ImportError as e:
            raise ImportError("ASE is not installed. Please install it to use this feature.") from e
        if not isinstance(atoms_obj, Atoms):
            raise TypeError("The 'atoms' field must be an instance of ase.Atoms.")
        return atoms_obj

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class DFTResult(BaseModel):
    job_id: UUID
    energy: float = Field(..., description="Final converged total energy in eV.")
    forces: list[list[float]] = Field(
        ..., description="Forces on each atom in eV/Angstrom. Shape: (N_atoms, 3)."
    )
    stress: list[float] = Field(
        ...,
        description="Virial stress tensor in Voigt notation (xx, yy, zz, yz, xz, xy) in eV/Angstrom^3.",
    )

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, forces: list[list[float]]) -> list[list[float]]:
        if not all(len(row) == 3 for row in forces):
            raise ValueError("Forces must have a shape of (N_atoms, 3).")
        return forces

    @field_validator("stress")
    @classmethod
    def check_stress_shape(cls, stress: list[float]) -> list[float]:
        if len(stress) != 6:
            raise ValueError("Stress tensor must have 6 components (Voigt notation).")
        return stress

    model_config = ConfigDict(extra="forbid")


class TrainingData(BaseModel):
    energy: float
    forces: list[list[float]]
    model_config = ConfigDict(extra="forbid")


# Pacemaker Models


class PacemakerLossWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy: float = Field(..., gt=0)
    forces: float = Field(..., gt=0)
    stress: float = Field(..., gt=0)


class PacemakerACEParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    radial_basis: str
    correlation_order: int = Field(..., ge=2)
    element_dependent_cutoffs: bool


class PacemakerFitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_filename: str
    loss_weights: PacemakerLossWeights
    ace: PacemakerACEParams


class PacemakerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fit_params: PacemakerFitParams


class CheckpointState(BaseModel):
    """
    Represents the serializable state of a workflow run for checkpointing.

    This model captures all the necessary information to resume a workflow
    from an interruption. It includes the original configuration, the current
    stage of the active learning loop, and, most importantly, the state of
    all submitted jobs. By storing the arguments needed to recreate a job
    rather than a non-serializable object like a Dask Future, the state can
    be reliably saved to and loaded from a file.
    """

    run_uuid: UUID
    system_config: SystemConfig
    active_learning_generation: int = Field(0, ge=0)
    current_potential_path: Path | None = None
    pending_job_ids: list[UUID] = Field(default_factory=list)
    job_submission_args: dict[UUID, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class CalculationMetadata(BaseModel):
    """Schema for metadata to be stored with each calculation in the database."""

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(
        ..., description="The name of the workflow stage that produced the structure."
    )
    uuid: str = Field(..., description="A unique identifier for the calculation run.")
    force_mask: list[list[float]] | None = Field(
        default=None,
        description="A per-atom mask to exclude forces from training.",
    )
