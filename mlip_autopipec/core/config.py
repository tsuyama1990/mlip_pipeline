from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field


class DFTConfig(BaseModel):
    """Configuration for DFT calculations."""

    model_config = ConfigDict(extra="forbid")

    code: Literal["quantum_espresso"] = Field("quantum_espresso", description="DFT code to use")
    command: str = Field(..., description="Command to execute the DFT code")
    pseudopotential_dir: DirectoryPath = Field(
        ..., description="Path to pseudopotentials directory"
    )
    ecutwfc: float = Field(40.0, gt=0, description="Wavefunction cutoff energy (Ry)")
    scf_convergence_threshold: float = Field(
        1e-6, gt=0, description="SCF convergence threshold (Ry)"
    )
    mixing_beta: float = Field(0.7, gt=0, le=1.0, description="Mixing beta for electron density")
    smearing: str = Field("mv", description="Smearing scheme")
    kpoints_density: float = Field(0.15, gt=0, description="K-points density (1/A)")


class GlobalConfig(BaseModel):
    """Global configuration for the MLIP pipeline."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(..., min_length=1, description="Name of the project")
    database_path: Path = Field(..., description="Path to the ASE database file")
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level"
    )


class AppConfig(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid")

    global_config: GlobalConfig = Field(..., alias="global")
    dft: DFTConfig


def load_config(path: Path) -> AppConfig:
    """Loads and validates the configuration from a YAML file."""
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        data = yaml.safe_load(f)

    # Check if data is None or not dict
    if not isinstance(data, dict):
        msg = "Invalid config file format. Must be a YAML dictionary."
        raise TypeError(msg)

    return AppConfig(**data)
