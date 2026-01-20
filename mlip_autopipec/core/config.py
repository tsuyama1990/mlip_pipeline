from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field


class DFTConfig(BaseModel):  # type: ignore
    """
    Configuration for the Density Functional Theory (DFT) calculations.
    """
    code: Literal["quantum_espresso"]
    command: str
    pseudopotential_dir: DirectoryPath
    scf_convergence_threshold: float = Field(default=1e-6, gt=0)
    mixing_beta: float = Field(default=0.7, gt=0, le=1.0)
    smearing: str = "mv"

    model_config = ConfigDict(extra="forbid")


class GlobalConfig(BaseModel):  # type: ignore
    """
    Global configuration for the MLIP-AutoPipe project.
    """
    project_name: str
    database_path: Path  # Using Path instead of FilePath to allow creation
    logging_level: Literal["DEBUG", "INFO", "WARNING"]

    model_config = ConfigDict(extra="forbid")
