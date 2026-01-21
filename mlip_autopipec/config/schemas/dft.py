from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DFTConfig(BaseModel):
    pseudopotential_dir: Path = Field(..., description="Directory containing .UPF files")
    ecutwfc: float = Field(..., gt=0, description="Wavefunction cutoff energy (Ry)")
    kspacing: float = Field(..., gt=0, description="Inverse K-point density (1/A)")
    nspin: int = Field(1, description="Spin polarization (1=off, 2=on)")

    model_config = ConfigDict(extra="forbid")

    @field_validator("pseudopotential_dir")
    @classmethod
    def validate_pseudo_dir(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"Pseudopotential directory does not exist: {v}")
        return v

    @field_validator("nspin")
    @classmethod
    def validate_nspin(cls, v: int) -> int:
        if v not in (1, 2):
            raise ValueError("nspin must be 1 or 2.")
        return v
