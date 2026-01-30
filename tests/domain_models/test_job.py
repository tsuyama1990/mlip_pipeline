from pathlib import Path

from mlip_autopipec.domain_models.job import JobStatus, JobResult, LammpsResult
from mlip_autopipec.domain_models.structure import Structure


def test_job_status_enum():
    assert JobStatus.PENDING == "PENDING"
    assert JobStatus.RUNNING == "RUNNING"
    assert JobStatus.COMPLETED == "COMPLETED"
    assert JobStatus.FAILED == "FAILED"
    assert JobStatus.TIMEOUT == "TIMEOUT"


def test_job_result_valid():
    result = JobResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=10.5,
        log_content="simulation done"
    )
    assert result.job_id == "123"
    assert result.status == JobStatus.COMPLETED
    assert result.duration_seconds == 10.5


def test_job_result_defaults():
    result = JobResult(
        job_id="123",
        status=JobStatus.PENDING,
        work_dir=Path("/tmp")
    )
    assert result.duration_seconds == 0.0
    assert result.log_content == ""


def test_lammps_result_valid(sample_ase_atoms):
    structure = Structure.from_ase(sample_ase_atoms)
    result = LammpsResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        final_structure=structure,
        trajectory_path=Path("/tmp/traj.xyz"),
        max_gamma=1.2
    )
    assert result.final_structure is not None
    assert result.final_structure.symbols == ["H", "H"]
    assert result.trajectory_path == Path("/tmp/traj.xyz")
    assert result.max_gamma == 1.2
