from pydantic import BaseModel, ConfigDict


class DFTParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pseudopotentials: dict[str, str]
    cutoff_wfc: float
    k_points: tuple[int, int, int]
    smearing_type: str
    degauss: float
    nspin: int


class GeneratorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generation_type: str
    sqs_supercell_size: list[int]
    strain_magnitudes: list[float]
    rattle_std_dev: float


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dft_params: DFTParams
    generator_params: GeneratorParams
