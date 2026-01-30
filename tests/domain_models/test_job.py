import pytest
from pathlib import Path
from pydantic import ValidationError
from mlip_autopipec.domain_models.job import JobResult, JobStatus

def test_job_result_valid():
    """Test creating a valid JobResult."""
    result = JobResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=10.5,
        log_content="simulation done"
    )
    assert result.job_id == "123"
    assert result.status == JobStatus.COMPLETED

def test_job_result_invalid_status():
    """Test that invalid status raises ValidationError."""
    with pytest.raises(ValidationError):
        JobResult(
            job_id="123",
            status="INVALID_STATUS", # type: ignore
            work_dir=Path("/tmp"),
            duration_seconds=10.5,
            log_content="log"
        )
