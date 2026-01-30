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
    """Base model for job execution results."""
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    work_dir: Path
    duration_seconds: float = 0.0
    log_content: str = ""


class LammpsResult(JobResult):
    """Result specific to a LAMMPS MD simulation."""
    final_structure: Optional[Structure] = None
    trajectory_path: Optional[Path] = None
    max_gamma: Optional[float] = None
