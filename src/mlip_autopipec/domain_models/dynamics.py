from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class MDStatus(str, Enum):
    COMPLETED = "completed"
    HALTED = "halted"
    FAILED = "failed"


class MDResult(BaseModel):
    status: MDStatus
    trajectory_path: Path | None = None
    log_path: Path | None = None
    halt_step: int | None = None
    final_structure_path: Path | None = None

    model_config = ConfigDict(extra="forbid")
