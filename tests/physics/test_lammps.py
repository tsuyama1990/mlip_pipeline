import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.domain_models import LammpsConfig, MDParams, PotentialConfig, ACEConfig
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
    return MDParams(
        temperature=300.0,
        n_steps=100,
        timestep=0.001,
        ensemble="NVT",
        uncertainty_threshold=5.0
    )


@pytest.fixture
def lammps_config():
    return LammpsConfig(command="lmp", timeout=10, use_mpi=False)


@pytest.fixture
def potential_config():
    return PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        zbl_inner_cutoff=0.5,
        zbl_outer_cutoff=1.2,
        ace_params=ACEConfig(
            npot="FinnisSinclair",
            fs_parameters=[1, 1, 1, 0.5],
            ndensity=2
        )
    )


def test_lammps_runner_execution(dummy_structure, md_params, lammps_config, potential_config, tmp_path):
    from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

    # Mock subprocess.Popen
    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "mlip_autopipec.physics.dynamics.lammps.LammpsRunner._parse_output"
        ) as mock_parse,
        patch("shutil.which") as mock_which,
        patch("select.select") as mock_select,
    ):
        # Setup mocks
        process_mock = MagicMock()
        process_mock.stdout.fileno.return_value = 1
        process_mock.stderr.fileno.return_value = 2
        process_mock.stdout.read.side_effect = [b"Simulation done", b"", b""]
        process_mock.stderr.read.side_effect = [b"", b"", b""]
        process_mock.poll.side_effect = [None, 0] # Running, then done
        process_mock.returncode = 0
        mock_popen.return_value = process_mock

        # Mock select to return ready FDs
        # It returns (rlist, wlist, xlist)
        # We return [1, 2] as ready
        mock_select.return_value = ([1, 2], [], [])

        mock_which.return_value = "/usr/bin/lmp"

        # Mock parsing to return the same structure
        mock_parse.return_value = (dummy_structure, Path("dump.lammpstrj"), None)

        runner = LammpsRunner(config=lammps_config, potential_config=potential_config, base_work_dir=tmp_path)
        potential_path = Path("potential.yace")
        result = runner.run(dummy_structure, md_params, potential_path=potential_path)

        assert result.status == JobStatus.COMPLETED, f"Job failed with log: {result.log_content}"
        assert result.final_structure == dummy_structure

        # Verify subprocess called with correct command
        # Expected: lmp -in in.lammps
        args, _ = mock_popen.call_args
        cmd = args[0]
        assert cmd[0] == "lmp"
        assert "-in" in cmd

        # Inspect generated input file
        # Find the job dir
        job_dirs = list(tmp_path.glob("job_*"))
        assert len(job_dirs) == 1
        work_dir = job_dirs[0]
        in_lammps = (work_dir / "in.lammps").read_text()

        # Check Hybrid Potential
        assert "pair_style" in in_lammps
        assert "hybrid/overlay pace zbl" in in_lammps
        assert "pair_coeff      * * pace" in in_lammps
        assert "Si" in in_lammps
        # Check for ZBL coefficients
        assert "pair_coeff" in in_lammps
        assert "zbl" in in_lammps

        # Check UQ
        assert "compute         pace_gamma all pace" in in_lammps
        assert "potential.yace" in in_lammps
        assert "fix             watchdog all halt 10 v_max_gamma > 5.0" in in_lammps


def test_lammps_runner_failure(dummy_structure, md_params, lammps_config, potential_config, tmp_path):
    from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

    with patch("subprocess.Popen") as mock_popen, patch("shutil.which") as mock_which, patch("select.select") as mock_select:

        process_mock = MagicMock()
        process_mock.stdout.fileno.return_value = 1
        process_mock.stderr.fileno.return_value = 2
        process_mock.stdout.read.return_value = b""
        process_mock.stderr.read.return_value = b"Error"
        process_mock.poll.return_value = 1
        process_mock.returncode = 1
        mock_popen.return_value = process_mock

        mock_select.return_value = ([1, 2], [], [])
        mock_which.return_value = "/usr/bin/lmp"

        runner = LammpsRunner(config=lammps_config, potential_config=potential_config, base_work_dir=tmp_path)
        result = runner.run(dummy_structure, md_params)

        assert result.status == JobStatus.FAILED
        assert "Error" in result.log_content
