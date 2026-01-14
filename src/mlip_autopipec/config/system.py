"""System-level configuration schemas for MLIP-AutoPipe.

These models represent the fully-expanded, internal configuration that drives
the entire workflow. They are produced from a `UserConfig` by a
heuristic engine and are considered read-only by all consumer modules.
"""

from pydantic import BaseModel, ConfigDict, Field


class DFTControlParams(BaseModel):
    """Parameters for the &CONTROL namelist in Quantum Espresso."""

    model_config = ConfigDict(extra="forbid")

    calculation: str = Field("scf", description="Type of calculation.")
    verbosity: str = Field("high", description="Level of output verbosity.")
    prefix: str = Field("dft_calc", description="Prefix for calculation files.")
    outdir: str = Field("./out", description="Output directory for run files.")


class DFTSystemParams(BaseModel):
    """Parameters for the &SYSTEM namelist in Quantum Espresso."""

    model_config = ConfigDict(extra="forbid")

    ibrav: int = Field(0, description="Bravais lattice index.")
    celldm: list[float] | None = Field(
        None, description="Cell dimensions (a, b/a, c/a, cos(ab), cos(ac), cos(bc))"
    )
    nat: int = Field(..., ge=1, description="Number of atoms in the system.")
    ntyp: int = Field(..., ge=1, description="Number of types of atoms.")
    ecutwfc: float = Field(
        60.0, description="Kinetic energy cutoff for wavefunctions in Ry."
    )
    ecutrho: float | None = Field(
        None, description="Kinetic energy cutoff for charge density in Ry."
    )
    occupations: str = Field(
        "smearing", description="Method for handling electronic occupations."
    )
    smearing: str = Field("mv", description="Type of smearing for metallic systems.")
    degauss: float = Field(0.01, description="Smearing width in Ry.")
    nspin: int = Field(1, description="Spin polarization (1=off, 2=on).")


class DFTElectronsParams(BaseModel):
    """Parameters for the &ELECTRONS namelist in Quantum Espresso."""

    model_config = ConfigDict(extra="forbid")

    mixing_beta: float = Field(0.7, description="Mixing factor for self-consistency.")
    conv_thr: float = Field(1.0e-10, description="Convergence threshold for SCF.")


class DFTParams(BaseModel):
    """Comprehensive container for all DFT-related parameters."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field("pw.x", description="The Quantum Espresso executable to run.")
    pseudopotentials: dict[str, str] = Field(
        ..., description="Mapping from element symbol to pseudopotential filename."
    )
    control: DFTControlParams = Field(
        default_factory=DFTControlParams, description="&CONTROL namelist parameters."
    )
    system: DFTSystemParams = Field(..., description="&SYSTEM namelist parameters.")
    electrons: DFTElectronsParams = Field(
        default_factory=DFTElectronsParams,
        description="&ELECTRONS namelist parameters.",
    )


class SystemConfig(BaseModel):
    """The root internal configuration object for an MLIP-AutoPipe workflow.

    This model is the single source of truth for all modules. It is created
    by a heuristic engine from a `UserConfig` and is treated as immutable
    by all subsequent components.
    """

    model_config = ConfigDict(extra="forbid")

    dft: DFTParams = Field(..., description="All parameters for the DFT engine.")
    db_path: str = Field(
        "mlip_database.db", description="Path to the central ASE database."
    )
