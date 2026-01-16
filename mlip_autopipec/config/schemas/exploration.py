from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

class SurrogateModelParams(BaseModel):
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
        import os
        if ".." in v or os.path.isabs(v):
            raise ValueError("model_path must be a relative path.")
        return v

class SOAPParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_max: int = Field(8)
    l_max: int = Field(6)
    r_cut: float = Field(5.0)
    atomic_sigma: float = Field(0.5)

class FPSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_structures_to_select: int = Field(200, ge=1)
    soap_params: SOAPParams = Field(default_factory=SOAPParams)

class ExplorerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    surrogate_model: SurrogateModelParams
    fps: FPSParams = Field(default_factory=FPSParams)

# Generator Models
class AlloyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sqs_supercell_size: list[int] = Field(default=[4, 4, 4], min_length=3)
    strain_magnitudes: list[float] = Field(default=[0.95, 1.0, 1.05])
    rattle_std_devs: list[float] = Field(default=[0.05, 0.1])

class CrystalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defect_types: list[Literal["vacancy", "interstitial"]] = Field(default=["vacancy"])

class GeneratorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alloy_params: AlloyParams = Field(default_factory=AlloyParams)
    crystal_params: CrystalParams = Field(default_factory=CrystalParams)
