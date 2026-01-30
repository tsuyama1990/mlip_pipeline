from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.structure import Structure


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class JobResult(BaseModel):
    """Base model for job results."""
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    work_dir: Path
    duration_seconds: float
    log_content: str = Field(description="Tail of the log content")


class LammpsResult(JobResult):
    """Result of a LAMMPS simulation."""
    final_structure: Optional[Structure] = None
    trajectory_path: Optional[Path] = None
    max_gamma: Optional[float] = None
