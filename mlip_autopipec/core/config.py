"""
Configuration schemas for MLIP-AutoPipe.
"""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mlip_autopipec.core.exceptions import ConfigError


class DFTConfig(BaseModel):
    """Configuration for DFT calculations."""
    code: Literal["quantum_espresso"] = Field(
        "quantum_espresso",
        description="The DFT code to use."
    )
    command: str = Field(
        ...,
        description="The command to run the DFT code (e.g., 'mpirun -np 4 pw.x')."
    )
    pseudopotential_dir: Path = Field(
        ...,
        description="Directory containing pseudopotentials."
    )
    scf_convergence_threshold: float = Field(
        1e-6,
        description="SCF convergence threshold in Ry.",
        gt=0
    )
    mixing_beta: float = Field(
        0.7,
        description="Mixing beta for electron density.",
        gt=0, le=1
    )
    smearing: str = Field(
        "mv",
        description="Smearing scheme (e.g., 'mv', 'gauss')."
    )
    kpoints_density: float = Field(
        0.15,
        description="K-points density for grid generation."
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def is_valid_pseudopotential_dir(self) -> bool:
        return self.pseudopotential_dir.is_dir()

class GlobalConfig(BaseModel):
    """Global configuration settings."""
    project_name: str = Field(..., description="Name of the project.")
    database_path: Path = Field(..., description="Path to the ASE database file.")
    logging_level: Literal["DEBUG", "INFO", "WARNING"] = Field(
        "INFO",
        description="Logging level."
    )

    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    """Root configuration object."""
    global_config: GlobalConfig = Field(..., alias="global")
    dft_config: DFTConfig = Field(..., alias="dft")

    model_config = ConfigDict(extra="forbid")


def load_config(path: Path) -> AppConfig:
    """
    Loads and validates the configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        AppConfig: The validated configuration object.

    Raises:
        ConfigError: If the configuration is invalid or file not found.
    """
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise ConfigError(msg)

    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML file: {e}"
        raise ConfigError(msg) from e
    except OSError as e:
        msg = f"Error reading configuration file: {e}"
        raise ConfigError(msg) from e

    if not data:
        msg = "Configuration file is empty"
        raise ConfigError(msg)

    try:
        config = AppConfig(**data)
    except ValidationError as e:
        msg = f"Configuration validation failed: {e}"
        raise ConfigError(msg) from e

    # Additional validation logic if needed (e.g., check paths exist)
    if not config.dft_config.pseudopotential_dir.is_dir():
            msg = f"Pseudopotential directory does not exist: {config.dft_config.pseudopotential_dir}"
            raise ConfigError(msg)

    # Create parent dir for database if it doesn't exist
    try:
        if not config.global_config.database_path.parent.exists():
            config.global_config.database_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"Failed to create database directory: {e}"
        raise ConfigError(msg) from e

    return config
