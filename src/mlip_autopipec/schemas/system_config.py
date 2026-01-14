from pydantic import BaseModel, ConfigDict

from .user_config import SurrogateConfig, TrainerConfig, UserConfig


class DFTParams(BaseModel):
    """Parameters for DFT calculations."""

    model_config = ConfigDict(extra="forbid")

    pseudopotentials: dict[str, str]
    cutoff_wfc: float
    k_points: tuple[int, int, int]
    smearing: str
    degauss: float
    nspin: int
    mixing_beta: float | None = None


class GeneratorParams(BaseModel):
    """Parameters for the structure generator."""

    model_config = ConfigDict(extra="forbid")

    sqs_supercell_size: list[int]
    strain_magnitudes: list[float]
    rattle_standard_deviation: float
    lattice_constant: float | None = None
    cutoffs: list[float] | None = None


class SystemConfig(BaseModel):
    """Fully specified system configuration."""

    model_config = ConfigDict(extra="forbid")

    user_config: UserConfig
    dft_params: DFTParams
    generator_params: GeneratorParams
    surrogate_config: SurrogateConfig
    trainer_config: TrainerConfig
