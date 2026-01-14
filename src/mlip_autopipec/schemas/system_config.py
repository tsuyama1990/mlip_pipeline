from pydantic import BaseModel, ConfigDict

from typing import List, Optional, Tuple

from .user_config import UserConfig


class DFTParams(BaseModel):
    """Parameters for DFT calculations."""

    model_config = ConfigDict(extra="forbid")

    pseudopotentials: dict[str, str]
    cutoff_wfc: float
    k_points: tuple[int, int, int]
    smearing_type: str
    degauss: float
    nspin: int
    mixing_beta: Optional[float] = None


class GeneratorParams(BaseModel):
    """Parameters for the structure generator."""

    model_config = ConfigDict(extra="forbid")

    sqs_supercell_size: list[int]
    strain_magnitudes: list[float]
    rattle_std_dev: float
    lattice_constant: Optional[float] = None
    cutoffs: Optional[List[float]] = None


class SystemConfig(BaseModel):
    """Fully specified system configuration."""

    model_config = ConfigDict(extra="forbid")

    user_config: UserConfig
    dft_params: DFTParams
    generator_params: GeneratorParams
