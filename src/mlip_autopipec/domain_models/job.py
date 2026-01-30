from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from mlip_autopipec.domain_models.structure import Structure


class JobStatus(str, Enum):
    """Status of an external calculation job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class JobResult(BaseModel):
    """
    Base model for the result of an external calculation.
    Capture metadata common to all HPC jobs.
    """
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    work_dir: Path
    duration_seconds: float
    log_content: str  # Tail of the log for quick debugging


class LammpsResult(JobResult):
    """
    Result of a LAMMPS MD simulation.
    """
    final_structure: Structure
    trajectory_path: Path
    max_gamma: Optional[float] = None
