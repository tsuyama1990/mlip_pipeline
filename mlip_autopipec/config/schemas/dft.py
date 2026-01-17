from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator

# --- Helper Models (Kept for InputGenerator usage, though not in SystemConfig directly anymore) ---


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
    model_config = ConfigDict(extra="forbid")

    @field_validator("k_points")
    @classmethod
    def k_points_must_be_positive(cls, k_points: tuple[int, int, int]) -> tuple[int, int, int]:
        if not all(k > 0 for k in k_points):
            raise ValueError("All k-point dimensions must be positive integers.")
        return k_points


# --- Cycle 02 Schema ---


class DFTConfig(BaseModel):
    """
    Configuration for the DFT Factory (Cycle 02).
    """

    command: str = "mpirun -np 4 pw.x"
    pseudo_dir: Path
    timeout: int = 3600
    recoverable: bool = True
    max_retries: int = 5

    # Optional Legacy Field for compatibility with Generator module
    dft_input_params: DFTInputParameters | None = None

    model_config = ConfigDict(extra="forbid")


class DFTJob(BaseModel):
    atoms: object
    params: DFTInputParameters  # We might still use this internally
    job_id: UUID = Field(default_factory=uuid4)
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("atoms")
    def validate_atoms_type(cls, atoms_obj: Any) -> Any:
        try:
            from ase import Atoms
        except ImportError as e:
            raise ImportError("ASE is not installed.") from e
        if not isinstance(atoms_obj, Atoms):
            raise TypeError("The 'atoms' field must be an instance of ase.Atoms.")
        return atoms_obj
