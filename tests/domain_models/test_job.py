from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.job import JobResult, JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure


def test_job_result_valid() -> None:
    """Test creating a valid JobResult."""
    res = JobResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=1.5,
        log_content="Finished"
    )
    assert res.status == "COMPLETED"
    assert res.duration_seconds == 1.5

def test_lammps_result_valid(minimal_structure: Structure) -> None:
    """Test creating a valid LammpsResult."""
    res = LammpsResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=1.5,
        log_content="Finished",
        final_structure=minimal_structure,
        trajectory_path=Path("traj.dump")
    )
    assert res.final_structure.symbols == minimal_structure.symbols

def test_job_result_invalid_status() -> None:
    """Test invalid status raises error."""
    with pytest.raises(ValidationError):
        JobResult(
            job_id="123",
            status="INVALID",  # type: ignore
            work_dir=Path("/tmp"),
            duration_seconds=1.0,
            log_content=""
        )
