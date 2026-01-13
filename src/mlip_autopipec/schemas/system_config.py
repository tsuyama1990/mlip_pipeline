"""Pydantic models for the internal, fully-specified system configuration."""

from pydantic import BaseModel, ConfigDict


class DFTParams(BaseModel):
    """Explicitly defines all parameters for a DFT calculation."""

    model_config = ConfigDict(extra="forbid")

    pseudopotentials: dict[str, str]
    cutoff_wfc: float
    k_points: tuple[int, int, int]
    smearing_type: str
    degauss: float
    nspin: int
    ecutrho: float
    mixing_beta: float
    diagonalization: str


class GeneratorParams(BaseModel):
    """Detailed parameters for the structure generator."""

    model_config = ConfigDict(extra="forbid")

    sqs_supercell_size: list[int]
    strain_magnitudes: list[float]
    rattle_std_dev: float
    eos_strain_magnitudes: list[float]


class SystemConfig(BaseModel):
    """The complete, expanded configuration for a single pipeline run."""

    model_config = ConfigDict(extra="forbid")

    project_name: str
    generation_type: str
    dft_params: DFTParams
    generator_params: GeneratorParams
