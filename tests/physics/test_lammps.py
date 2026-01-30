from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.domain_models.config import LammpsConfig
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner, MDParams


@pytest.fixture
def minimal_structure(sample_ase_atoms):
    return Structure.from_ase(sample_ase_atoms)


@pytest.fixture
def lammps_config():
    return LammpsConfig(command="lmp_mock", timeout=10, cores=1)


@pytest.fixture
def runner(lammps_config):
    return LammpsRunner(lammps_config)


@pytest.fixture
def md_params():
    return MDParams(temperature=300, n_steps=100)


def test_runner_init(runner):
    assert runner.config.command == "lmp_mock"


@patch("subprocess.run")
def test_run_mocked_success(mock_run, runner, minimal_structure, md_params, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stdout="Simulation done", stderr="")

    # We mock _write_inputs and _parse_output to isolate run logic?
    # No, we want to test that they are called.
    # But _parse_output reads file. We need to mock file existence or _parse_output.

    # Mock _parse_output
    with patch.object(runner, '_parse_output') as mock_parse:
        mock_parse.return_value = LammpsResult(
            job_id="test", status=JobStatus.COMPLETED, work_dir=tmp_path, duration_seconds=1.0, log_content=""
        )

        result = runner.run(minimal_structure, tmp_path, md_params)

        assert result.status == JobStatus.COMPLETED
        # Verify subprocess called
        mock_run.assert_called()


def test_md_params_validation():
    p = MDParams(temperature=300)
    assert p.ensemble == "nvt"
