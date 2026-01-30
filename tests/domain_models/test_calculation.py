from pathlib import Path
import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult
import numpy as np
from mlip_autopipec.domain_models.job import JobStatus


def test_dft_config_valid():
    config = DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.upf")},
        ecutwfc=40.0,
        kspacing=0.04
    )
    assert config.command == "pw.x"
    assert config.kspacing == 0.04


def test_dft_config_invalid_extra_field():
    with pytest.raises(ValidationError):
        DFTConfig(
            command="pw.x",
            pseudopotentials={"Si": Path("Si.upf")},
            ecutwfc=40.0,
            extra_field="invalid"
        )

def test_dft_result_valid():
    result = DFTResult(
        job_id="test_job",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=10.0,
        log_content="success",
        energy=-100.0,
        forces=np.zeros((2, 3)),
        stress=np.eye(3)
    )
    assert result.energy == -100.0
    assert result.forces.shape == (2, 3)
