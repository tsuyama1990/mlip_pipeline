from pathlib import Path
import numpy as np
from mlip_autopipec.domain_models.job import JobResult, JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure


def test_job_status_enum():
    assert JobStatus.PENDING == "PENDING"
    assert JobStatus.COMPLETED == "COMPLETED"


def test_job_result_valid():
    res = JobResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=10.5,
        log_content="Finished"
    )
    assert res.job_id == "123"
    assert res.status == JobStatus.COMPLETED


def test_lammps_result_valid():
    struct = Structure(
        symbols=["Si"],
        positions=np.array([[0,0,0]]),
        cell=np.eye(3),
        pbc=(True, True, True)
    )
    res = LammpsResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=10.5,
        final_structure=struct,
        trajectory_path=Path("traj.dump")
    )
    assert res.final_structure is not None
    assert res.trajectory_path == Path("traj.dump")


def test_lammps_result_optional():
    res = LammpsResult(
        job_id="123",
        status=JobStatus.FAILED,
        work_dir=Path("/tmp"),
        duration_seconds=1.0
    )
    assert res.final_structure is None
