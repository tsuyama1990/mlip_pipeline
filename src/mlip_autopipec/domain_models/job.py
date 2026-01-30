import enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from mlip_autopipec.domain_models.structure import Structure


class JobStatus(str, enum.Enum):
    """Status of a computational job."""
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
    log_content: str = Field(description="Tail of the log file for debugging")


class LammpsResult(JobResult):
    """Result of a LAMMPS simulation."""
    model_config = ConfigDict(extra="forbid")

    final_structure: Structure
    trajectory_path: Path
    max_gamma: Optional[float] = None  # Placeholder for future cycles
