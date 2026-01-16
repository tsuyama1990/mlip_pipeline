# FIXME: The above comment is a temporary workaround for a ruff bug.
# It should be removed once the bug is fixed.
# For more information, see: https://github.com/astral-sh/ruff/issues/10515
from typing import Literal

"""
This module defines the Pydantic models that constitute the core data
structures and configuration for the MLIP-AutoPipe workflow.

The "Schema-First" design principle is strictly enforced here. These models
are the single source of truth for the application's data, ensuring type
safety, validation, and clear data contracts between different components.
"""
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    RootModel,
    ValidationInfo,
    field_validator,
)


class CutoffConfig(BaseModel):
    """Configuration for plane-wave and charge density cutoffs."""

    wavefunction: float = Field(
        ...,
        description="Plane-wave cutoff energy in Rydberg.",
        gt=0,
    )
    density: float = Field(
        ...,
        description="Charge density cutoff energy in Rydberg.",
        gt=0,
    )
    model_config = ConfigDict(extra="forbid")


class FingerprintConfig(BaseModel):
    """Configuration for generating structural fingerprints."""

    type: Literal["soap"] = "soap"
    soap_rcut: float = Field(5.0, gt=0)
    soap_nmax: int = Field(8, gt=0)
    soap_lmax: int = Field(6, gt=0)
    species: list[str] = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")


class ExplorerConfig(BaseModel):
    """Configuration for the SurrogateExplorer."""

    surrogate_model_path: str  # Pydantic's FilePath is too strict for remote paths
    max_force_threshold: float = Field(10.0, gt=0)
    fingerprint: FingerprintConfig
    model_config = ConfigDict(extra="forbid")


class SmearingConfig(BaseModel):
    """Configuration for metallic smearing."""

    smearing_type: Literal["gauss", "mv", "marzari-vanderbilt"] = Field(
        "mv",
        description="Type of smearing to be used for metals.",
    )
    degauss: float = Field(
        0.02,
        description="Smearing width in Rydberg.",
        gt=0,
    )
    model_config = ConfigDict(extra="forbid")


class MagnetismConfig(BaseModel):
    """Configuration for spin-polarized (magnetic) calculations."""

    nspin: Literal[2] = 2
    starting_magnetization: dict[str, float] = Field(
        ...,
        description="Initial magnetic moment for each atomic species.",
    )
    model_config = ConfigDict(extra="forbid")


class Pseudopotentials(RootModel[dict[str, str]]):
    """A mapping from chemical symbols to pseudopotential file names."""

    @field_validator("root")
    def validate_symbols(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that the keys are valid chemical symbols."""
        from ase.data import chemical_symbols

        for symbol in v:
            if symbol not in chemical_symbols:
                raise ValueError(f"'{symbol}' is not a valid chemical symbol.")
        return v


class DFTInputParameters(BaseModel):
    """
    A complete and validated set of parameters for a single Quantum Espresso
    'scf' calculation.

    This model ensures that a calculation is only attempted if all required
    parameters are present and self-consistent.
    """

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
    def k_points_must_be_positive(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate that k-point dimensions are positive."""
        if not all(k > 0 for k in v):
            raise ValueError("All k-point dimensions must be positive integers.")
        return v

    model_config = ConfigDict(extra="forbid")


class MDConfig(BaseModel):
    """Configuration for the molecular dynamics simulation."""

    ensemble: Literal["nvt", "npt"] = "nvt"
    temperature: float = Field(300.0, gt=0)
    timestep: float = Field(1.0, gt=0)
    run_duration: int = Field(1000, gt=0)
    model_config = ConfigDict(extra="forbid")


class UncertaintyConfig(BaseModel):
    """Configuration for the active learning trigger."""

    threshold: float = Field(5.0, gt=0)
    embedding_cutoff: float = Field(8.0, gt=0)
    masking_cutoff: float = Field(5.0, gt=0)
    model_config = ConfigDict(extra="forbid")

    @field_validator("masking_cutoff")
    @classmethod
    def masking_must_be_less_than_embedding(cls, v: float, info: ValidationInfo) -> float:
        """Validate that masking_cutoff is smaller than embedding_cutoff."""
        if "embedding_cutoff" in info.data and v >= info.data["embedding_cutoff"]:
            raise ValueError("masking_cutoff must be smaller than embedding_cutoff.")
        return v


class InferenceConfig(BaseModel):
    """Top-level configuration for the LammpsRunner."""

    lammps_executable: FilePath
    potential_path: FilePath
    md_params: MDConfig = Field(default_factory=MDConfig)
    uncertainty_params: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    model_config = ConfigDict(extra="forbid")


class UncertainStructure(BaseModel):
    """Data transfer object for a structure with high uncertainty."""

    atoms: object
    force_mask: object
    metadata: dict[str, Any] = {}
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("atoms")
    @classmethod
    def validate_atoms_type(cls, v: Any) -> Any:
        """Ensure 'atoms' is an ASE Atoms object."""
        try:
            from ase import Atoms
        except ImportError as e:
            raise ImportError("ASE is required for this model.") from e
        if not isinstance(v, Atoms):
            raise TypeError("Field 'atoms' must be an instance of ase.Atoms.")
        return v

    @field_validator("force_mask")
    @classmethod
    def validate_force_mask(cls, v: Any, info: ValidationInfo) -> Any:
        """Ensure 'force_mask' is a NumPy array with the correct shape."""
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("NumPy is required for this model.") from e
        if not isinstance(v, np.ndarray):
            raise TypeError("Field 'force_mask' must be a NumPy array.")
        if "atoms" in info.data and len(v) != len(info.data["atoms"]):
            raise ValueError("force_mask must have the same length as the number of atoms.")
        return v


class DFTJob(BaseModel):
    """
    Represents a single, self-contained DFT job to be executed.

    This model bundles the atomic structure (`ase.Atoms`) with its
    corresponding validated input parameters.
    """

    atoms: object  # Using `object` to prevent circular imports with `ase.Atoms`
    params: DFTInputParameters
    job_id: UUID = Field(default_factory=uuid4)

    @field_validator("atoms")
    def validate_atoms_type(cls, v):
        """Validate that the atoms object is a valid ase.Atoms instance."""
        try:
            from ase import Atoms
        except ImportError as e:
            raise ImportError("ASE is not installed. Please install it to use this feature.") from e

        if not isinstance(v, Atoms):
            raise TypeError("The 'atoms' field must be an instance of ase.Atoms.")
        return v

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class DFTResult(BaseModel):
    """
    Represents the structured output of a successful DFT calculation.

    This is the primary data transfer object returned by the DFTFactory,
    ensuring that consumers receive data in a consistent and validated format.
    """

    job_id: UUID
    energy: float = Field(..., description="Final converged total energy in eV.")
    forces: list[list[float]] = Field(
        ...,
        description="Forces on each atom in eV/Angstrom. Shape: (N_atoms, 3).",
    )
    stress: list[float] = Field(
        ...,
        description=(
            "Virial stress tensor in Voigt notation (xx, yy, zz, yz, xz, xy) in eV/Angstrom^3."
        ),
    )

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate that the forces array has the correct Nx3 shape."""
        if not all(len(row) == 3 for row in v):
            raise ValueError("Forces must have a shape of (N_atoms, 3).")
        return v

    @field_validator("stress")
    @classmethod
    def check_stress_shape(cls, v: list[float]) -> list[float]:
        """Validate that the stress tensor has 6 components (Voigt)."""
        if len(v) != 6:
            raise ValueError("Stress tensor must have 6 components (Voigt notation).")
        return v

    model_config = ConfigDict(extra="forbid")
