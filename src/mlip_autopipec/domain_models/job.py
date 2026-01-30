from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


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
