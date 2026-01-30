import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.domain_models import LammpsConfig, MDParams
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.job import JobStatus
import numpy as np


@pytest.fixture
def dummy_structure():
    return Structure(
        symbols=["Si", "Si"],
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
        cell=np.eye(3) * 5.43,
        pbc=(True, True, True),
    )


@pytest.fixture
def md_params():
    return MDParams(temperature=300.0, n_steps=100, timestep=0.001, ensemble="NVT")


@pytest.fixture
def lammps_config():
    return LammpsConfig(command="lmp_serial", timeout=10, use_mpi=False)


def test_lammps_runner_execution(dummy_structure, md_params, lammps_config, tmp_path):
    from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

    # Mock subprocess.run
    with (
        patch("subprocess.run") as mock_run,
        patch(
            "mlip_autopipec.physics.dynamics.lammps.LammpsRunner._parse_output"
        ) as mock_parse,
        patch("shutil.which") as mock_which,
    ):
        # Setup mocks
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Simulation done", stderr=""
        )
        mock_which.return_value = "/usr/bin/lmp_serial"

        # Mock parsing to return the same structure
        mock_parse.return_value = (dummy_structure, Path("dump.lammpstrj"))

        runner = LammpsRunner(config=lammps_config, base_work_dir=tmp_path)
        result = runner.run(dummy_structure, md_params)

        assert result.status == JobStatus.COMPLETED
        assert result.final_structure == dummy_structure

        # Verify subprocess called with correct command
        # Expected: lmp_serial -in in.lammps
        args, _ = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "lmp_serial"
        assert "-in" in cmd


def test_lammps_runner_failure(dummy_structure, md_params, lammps_config, tmp_path):
    from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

    with patch("subprocess.run") as mock_run, patch("shutil.which") as mock_which:
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "lmp_serial", stderr="Error"
        )
        mock_which.return_value = "/usr/bin/lmp_serial"

        runner = LammpsRunner(config=lammps_config, base_work_dir=tmp_path)
        result = runner.run(dummy_structure, md_params)

        assert result.status == JobStatus.FAILED
        assert "Error" in result.log_content
