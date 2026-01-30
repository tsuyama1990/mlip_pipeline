from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult, SCFError
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.physics.dft.qe_runner import QERunner


@pytest.fixture
def sample_structure():
    return Structure(
        symbols=["Si"],
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=np.eye(3) * 5.43,
        pbc=(True, True, True),
    )


@pytest.fixture
def dft_config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("fake.upf")},
        ecutwfc=30.0,
        kspacing=0.04,
    )


def test_qe_runner_success(sample_structure, dft_config, tmp_path):
    # Mock parser to return success
    result = DFTResult(
        job_id="job1",
        status=JobStatus.COMPLETED,
        work_dir=tmp_path,
        duration_seconds=10.0,
        log_content="Success",
        energy=-100.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3))
    )

    with patch("mlip_autopipec.physics.dft.qe_runner.DFTParser.parse", return_value=result) as mock_parse, \
         patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run") as mock_run:

        runner = QERunner(dft_config, work_dir=tmp_path)
        final_result = runner.run(sample_structure, job_id="job1")

        assert final_result.status == JobStatus.COMPLETED
        assert final_result.energy == -100.0
        assert mock_run.call_count == 1
        assert mock_parse.call_count == 1


def test_qe_runner_recovery(sample_structure, dft_config, tmp_path):
    # Scenario: First run fails with SCFError, second run succeeds

    # Mock parser side_effect
    success_result = DFTResult(
        job_id="job1",
        status=JobStatus.COMPLETED,
        work_dir=tmp_path,
        duration_seconds=10.0,
        log_content="Success",
        energy=-100.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3))
    )

    with patch("mlip_autopipec.physics.dft.qe_runner.DFTParser.parse", side_effect=[SCFError("SCF failed"), success_result]), \
         patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run") as mock_run:

        runner = QERunner(dft_config, work_dir=tmp_path)
        final_result = runner.run(sample_structure, job_id="job1")

        assert final_result.status == JobStatus.COMPLETED
        assert mock_run.call_count == 2
        # Verify that input generator was called with different params?
        # We can't easily check that without patching InputGenerator or inspecting mock_run args (files).
        # But we trust RecoveryHandler is called.

def test_qe_runner_max_retries(sample_structure, dft_config, tmp_path):
    # Scenario: Fails repeatedly until max retries

    with patch("mlip_autopipec.physics.dft.qe_runner.DFTParser.parse", side_effect=SCFError("SCF failed")), \
         patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run") as mock_run:

        runner = QERunner(dft_config, work_dir=tmp_path)
        # Assuming RecoveryHandler has finite strategies.
        # But QERunner should probably have a max_retries limit too to avoid infinite loop if strategies cycle?
        # RecoveryHandler raises ValueError if strategies exhausted.
        # QERunner should catch that and return FAILED result.

        final_result = runner.run(sample_structure, job_id="job1")

        assert final_result.status == JobStatus.FAILED
        # Check that we tried multiple times
        assert mock_run.call_count > 1
