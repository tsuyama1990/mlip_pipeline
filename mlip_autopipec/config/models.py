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


class DFTConfig(BaseModel):
    """Configuration for the DFT Factory."""

    # This is a placeholder for the full DFTConfig
    # For now, it just contains the input parameters
    dft_input_params: DFTInputParameters
    model_config = ConfigDict(extra="forbid")


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


class TrainingConfig(BaseModel):
    pacemaker_executable: FilePath | None = None
    data_source_db: Path
    template_file: FilePath | None = None
    delta_learning: bool = True
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    ace_params: "PacemakerACEParams" = Field(
        default_factory=lambda: PacemakerACEParams(
            radial_basis="radial", correlation_order=3, element_dependent_cutoffs=False
        )
    )
    model_config = ConfigDict(extra="forbid")


class WorkflowConfig(BaseModel):
    checkpoint_filename: str = "checkpoint.json"
    model_config = ConfigDict(extra="forbid")


class SystemConfig(BaseModel):
    project_name: str
    run_uuid: UUID
    workflow_config: WorkflowConfig = Field(default_factory=WorkflowConfig)
    dft_config: DFTConfig
    explorer_config: ExplorerConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    model_config = ConfigDict(extra="forbid")


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
