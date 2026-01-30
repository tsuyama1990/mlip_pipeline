from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.job import JobResult, JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def minimal_structure(sample_ase_atoms):
    return Structure.from_ase(sample_ase_atoms)


def test_job_status_enum():
    assert JobStatus.PENDING == "PENDING"
    assert JobStatus.RUNNING == "RUNNING"
    assert JobStatus.COMPLETED == "COMPLETED"
    assert JobStatus.FAILED == "FAILED"
    assert JobStatus.TIMEOUT == "TIMEOUT"


def test_job_result_valid():
    job = JobResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=10.5,
        log_content="simulation done"
    )
    assert job.job_id == "123"
    assert job.status == JobStatus.COMPLETED
    assert job.duration_seconds == 10.5


def test_job_result_invalid():
    with pytest.raises(ValidationError):
        JobResult(
            job_id="123",
            status="UNKNOWN",  # Invalid status
            work_dir=Path("/tmp"),
            duration_seconds=10.5,
            log_content="log"
        )


def test_lammps_result_valid(minimal_structure):
    res = LammpsResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=10.5,
        log_content="log",
        final_structure=minimal_structure,
        trajectory_path=Path("/tmp/traj.lammpstrj"),
        max_gamma=0.5
    )
    assert res.final_structure is not None
    assert res.max_gamma == 0.5


def test_lammps_result_optional_fields():
    res = LammpsResult(
        job_id="123",
        status=JobStatus.FAILED,
        work_dir=Path("/tmp"),
        duration_seconds=1.0,
        log_content="error"
    )
    assert res.final_structure is None
    assert res.trajectory_path is None
