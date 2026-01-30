from typing import Literal, Optional
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.domain_models.structure import Structure


class LammpsConfig(BaseModel):
    """Configuration for the LAMMPS executable and runtime environment."""

    model_config = ConfigDict(extra="forbid")

    command: str = "lmp_serial"
    timeout: int = 3600
    use_mpi: bool = False
    mpi_command: str = "mpirun -np 4"


class MDConfig(BaseModel):
    """Configuration for MD simulation parameters."""

    model_config = ConfigDict(extra="forbid")

    temperature: float
    pressure: Optional[float] = None
    n_steps: int
    timestep: float = 0.001
    ensemble: Literal["NVT", "NPT"]


# Alias for backward compatibility
MDParams = MDConfig


class LammpsResult(JobResult):
    """
    Result of a LAMMPS MD simulation.
    Includes trajectory path and maximum uncertainty (gamma) if applicable.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    final_structure: Structure
    trajectory_path: Path
    max_gamma: Optional[float] = None
