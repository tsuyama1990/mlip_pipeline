import numpy as np
import pytest
from pydantic import ValidationError
from pathlib import Path

from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult
from mlip_autopipec.domain_models.job import JobStatus


def test_dft_config_valid():
    config = DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.upf")},
        ecutwfc=40.0,
        kspacing=0.04,
    )
    assert config.ecutwfc == 40.0
    assert config.mixing_beta == 0.7  # default


def test_dft_config_invalid_ecutwfc():
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotentials={"Si": Path("Si.upf")},
            ecutwfc=-10.0,
            kspacing=0.04,
        )


def test_dft_config_invalid_mixing_beta():
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotentials={"Si": Path("Si.upf")},
            ecutwfc=40.0,
            kspacing=0.04,
            mixing_beta=1.5,
        )


def test_dft_result_valid():
    result = DFTResult(
        job_id="job_123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp/work"),
        duration_seconds=10.0,
        log_content="Finished",
        energy=-100.0,
        forces=np.zeros((2, 3)),
        stress=np.eye(3),
    )
    assert result.energy == -100.0
    assert result.forces.shape == (2, 3)


def test_dft_result_invalid_forces_shape():
    with pytest.raises(ValidationError):
        DFTResult(
            job_id="job_123",
            status=JobStatus.COMPLETED,
            work_dir=Path("/tmp/work"),
            duration_seconds=10.0,
            log_content="Finished",
            energy=-100.0,
            forces=np.zeros((2, 4)),  # Wrong shape
        )
