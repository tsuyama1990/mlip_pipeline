import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.domain_models.config import LammpsConfig, MDParams, PotentialConfig
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


@pytest.fixture
def potential_config():
    return PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        seed=42,
        pair_style="lj/cut 2.5",  # Explicitly set for this test
    )


def test_lammps_runner_execution(
    dummy_structure, md_params, lammps_config, potential_config, tmp_path
):
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

        # Update config with tmp_path
        lammps_config.base_work_dir = tmp_path
        runner = LammpsRunner(config=lammps_config, potential_config=potential_config)
        result = runner.run(dummy_structure, md_params)

        assert result.status == JobStatus.COMPLETED
        assert result.final_structure == dummy_structure

        # Verify subprocess called with correct command
        # Expected: lmp_serial -in in.lammps
        args, _ = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "lmp_serial"
        assert "-in" in cmd

        # Verify in.lammps content has pair_style
        # We need to check file content in tmp_path/job_.../in.lammps
        # Since job id is random, we search dir
        job_dir = next(tmp_path.glob("job_*"))
        in_file = job_dir / "in.lammps"
        assert "pair_style      lj/cut 2.5" in in_file.read_text()


def test_lammps_runner_hybrid_potential(
    dummy_structure, md_params, lammps_config, tmp_path
):
    from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

    hybrid_config = PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        seed=42,
        pair_style="hybrid/overlay ace zbl",
        ace_potential_file=Path("test.yace"),
    )

    with (
        patch("subprocess.run") as mock_run,
        patch(
            "mlip_autopipec.physics.dynamics.lammps.LammpsRunner._parse_output"
        ) as mock_parse,
        patch("shutil.which") as mock_which,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_which.return_value = "lmp"
        mock_parse.return_value = (dummy_structure, Path("dump.lammpstrj"))

        lammps_config.base_work_dir = tmp_path
        runner = LammpsRunner(config=lammps_config, potential_config=hybrid_config)
        runner.run(dummy_structure, md_params)

        job_dir = next(tmp_path.glob("job_*"))
        in_file = job_dir / "in.lammps"
        content = in_file.read_text()

        assert "pair_style      hybrid/overlay ace zbl" in content
        assert "pair_coeff * * pace test.yace Si" in content
        # ZBL: Si is 14
        # pair_coeff 1 1 zbl 14 14
        assert "pair_coeff 1 1 zbl 14 14" in content


def test_lammps_runner_failure(
    dummy_structure, md_params, lammps_config, potential_config, tmp_path
):
    from mlip_autopipec.physics.dynamics.lammps import LammpsRunner

    with patch("subprocess.run") as mock_run, patch("shutil.which") as mock_which:
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "lmp_serial", stderr="Error"
        )
        mock_which.return_value = "/usr/bin/lmp_serial"

        lammps_config.base_work_dir = tmp_path
        runner = LammpsRunner(config=lammps_config, potential_config=potential_config)
        result = runner.run(dummy_structure, md_params)

        assert result.status == JobStatus.FAILED
        assert "Error" in result.log_content
