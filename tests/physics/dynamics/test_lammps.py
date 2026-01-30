import subprocess
from unittest.mock import patch

import pytest

from mlip_autopipec.domain_models.config import LammpsConfig, MDParams, PotentialConfig, ElementParams
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock:
        yield mock

@pytest.fixture
def lammps_runner():
    config = LammpsConfig(command="echo 'mock'", timeout=10)
    return LammpsRunner(config)

@pytest.fixture
def potential_config():
    # Helper for valid potential config
    return PotentialConfig(
        elements=["H"],
        cutoff=2.5,
        element_params={
            "H": ElementParams(mass=1.008, lj_sigma=1.0, lj_epsilon=1.0, zbl_z=1)
        }
    )

def test_md_params_valid() -> None:
    """Test MDParams validation."""
    p = MDParams(temperature=300, n_steps=1000, timestep=0.001)
    assert p.temperature == 300

def test_run_success(lammps_runner, minimal_structure, potential_config, mock_subprocess_run, tmp_path):
    """Test successful run."""
    # Mock subprocess success
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Done", stderr=""
    )

    # Easiser: Mock the `_parse_output` method to return a structure.
    with patch.object(LammpsRunner, "_parse_output") as mock_parse:
        mock_parse.return_value = minimal_structure

        params = MDParams(temperature=300, n_steps=100)
        result = lammps_runner.run(minimal_structure, params, potential_config, work_dir=tmp_path)

        assert result.status == JobStatus.COMPLETED
        assert result.final_structure == minimal_structure

        # Verify inputs written
        assert (tmp_path / "in.lammps").exists()
        assert (tmp_path / "data.lammps").exists()

        # Verify hybrid potential input
        script = (tmp_path / "in.lammps").read_text()
        assert "pair_style      hybrid/overlay" in script
        assert "pair_coeff      1 1 zbl 1 1" in script # Z=1 for H
        assert "mass            1 1.008" in script

        # Verify command called
        mock_subprocess_run.assert_called_once()

def test_run_failed_subprocess(lammps_runner, minimal_structure, potential_config, mock_subprocess_run, tmp_path):
    """Test subprocess failure."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Error"
    )

    params = MDParams(temperature=300, n_steps=100)
    result = lammps_runner.run(minimal_structure, params, potential_config, work_dir=tmp_path)

    assert result.status == JobStatus.FAILED
    assert "Error" in result.log_content

def test_run_timeout(lammps_runner, minimal_structure, potential_config, mock_subprocess_run, tmp_path):
    """Test timeout."""
    mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="lmp", timeout=10)

    params = MDParams(temperature=300, n_steps=100)
    result = lammps_runner.run(minimal_structure, params, potential_config, work_dir=tmp_path)

    assert result.status == JobStatus.TIMEOUT

def test_run_missing_element_params(lammps_runner, minimal_structure, mock_subprocess_run, tmp_path):
    """Test failure if element params are missing."""
    bad_config = PotentialConfig(elements=["H"], cutoff=2.5) # No params
    params = MDParams(temperature=300)

    result = lammps_runner.run(minimal_structure, params, bad_config, work_dir=tmp_path)

    assert result.status == JobStatus.FAILED
    assert "Missing element parameters" in result.log_content


def test_parse_output(lammps_runner, tmp_path):
    """Test parsing a real dump file."""
    dump_file = tmp_path / "dump.lammpstrj"
    dump_content = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 5.43
0.0 5.43
0.0 5.43
ITEM: ATOMS id type x y z
1 1 0.1 0.1 0.1
2 1 1.5 1.5 1.5
"""
    dump_file.write_text(dump_content)

    struct = lammps_runner._parse_output(dump_file)
    assert len(struct.positions) == 2
    assert struct.cell[0, 0] == 5.43

def test_parse_output_missing(lammps_runner, tmp_path):
    with pytest.raises(FileNotFoundError):
        lammps_runner._parse_output(tmp_path / "missing.dump")
