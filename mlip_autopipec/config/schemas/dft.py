from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


class CutoffConfig(BaseModel):
    wavefunction: float = Field(..., description="Plane-wave cutoff energy in Rydberg.", gt=0)
    density: float = Field(..., description="Charge density cutoff energy in Rydberg.", gt=0)
    model_config = ConfigDict(extra="forbid")

class SmearingConfig(BaseModel):
    smearing_type: Literal["gauss", "mv", "marzari-vanderbilt"] = Field(
        "mv", description="Type of smearing to be used for metals."
    )
    degauss: float = Field(0.02, description="Smearing width in Rydberg.", gt=0)
    model_config = ConfigDict(extra="forbid")

class StartingMagnetization(RootModel[dict[str, float]]):
    @field_validator("root")
    @classmethod
    def validate_symbols(cls, mag_values: dict[str, str]) -> dict[str, str]:
        from ase.data import chemical_symbols
        for symbol in mag_values:
            if symbol not in chemical_symbols:
                msg = f"'{symbol}' is not a valid chemical symbol."
                raise ValueError(msg)
        return mag_values

class MagnetismConfig(BaseModel):
    nspin: Literal[2] = 2
    starting_magnetization: StartingMagnetization = Field(
        ..., description="Initial magnetic moment for each atomic species."
    )
    model_config = ConfigDict(extra="forbid")

class Pseudopotentials(RootModel[dict[str, str]]):
    @field_validator("root")
    @classmethod
    def validate_symbols(cls, pseudo_values: dict[str, str]) -> dict[str, str]:
        from ase.data import chemical_symbols
        for symbol in pseudo_values:
            if symbol not in chemical_symbols:
                msg = f"'{symbol}' is not a valid chemical symbol."
                raise ValueError(msg)
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
    model_config = ConfigDict(extra="forbid")

    @field_validator("k_points")
    @classmethod
    def k_points_must_be_positive(cls, k_points: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(k_points) != 3:
            msg = "k_points must be a tuple of exactly three integers."
            raise ValueError(msg)
        if not all(k > 0 for k in k_points):
            msg = "All k-point dimensions must be positive integers."
            raise ValueError(msg)
        return k_points

class DFTConfig(BaseModel):
    """Configuration for the DFT Factory."""
    dft_input_params: DFTInputParameters
    model_config = ConfigDict(extra="forbid")

class DFTJob(BaseModel):
    atoms: object
    params: DFTInputParameters
    job_id: UUID = Field(default_factory=uuid4)
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("atoms")
    @classmethod
    def validate_atoms_type(cls, atoms_obj: Any) -> Any:
        try:
            from ase import Atoms
        except ImportError as e:
            msg = "ASE is not installed."
            raise ImportError(msg) from e
        if not isinstance(atoms_obj, Atoms):
            msg = "The 'atoms' field must be an instance of ase.Atoms."
            raise TypeError(msg)
        return atoms_obj

class DFTResult(BaseModel):
    job_id: UUID
    energy: float = Field(..., description="Final converged total energy in eV.")
    forces: list[list[float]] = Field(
        ..., description="Forces on each atom in eV/Angstrom. Shape: (N_atoms, 3)."
    )
    stress: list[float] = Field(
        ...,
        description="Virial stress tensor in Voigt notation."
    )
    model_config = ConfigDict(extra="forbid")

    @field_validator("forces")
    @classmethod
    def check_forces_shape(cls, forces: list[list[float]]) -> list[list[float]]:
        if not all(len(row) == 3 for row in forces):
            msg = "Forces must have a shape of (N_atoms, 3)."
            raise ValueError(msg)
        return forces

    @field_validator("stress")
    @classmethod
    def check_stress_shape(cls, stress: list[float]) -> list[float]:
        if len(stress) != 6:
            msg = "Stress tensor must have 6 components (Voigt notation)."
            raise ValueError(msg)
        return stress
