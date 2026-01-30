import subprocess
from unittest.mock import patch

import pytest

from mlip_autopipec.domain_models.config import LammpsConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner, MDParams


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock:
        yield mock

@pytest.fixture
def lammps_runner():
    config = LammpsConfig(command="echo 'mock'", timeout=10)
    return LammpsRunner(config)

def test_md_params_valid() -> None:
    """Test MDParams validation."""
    p = MDParams(temperature=300, n_steps=1000, timestep=0.001)
    assert p.temperature == 300

def test_run_success(lammps_runner, minimal_structure, mock_subprocess_run, tmp_path):
    """Test successful run."""
    # Mock subprocess success
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Done", stderr=""
    )

    # Mock the output file creation (dump.lammpstrj)
    # We need to spy on the internal work_dir creation or mock internal methods
    # But integration test logic: wrapper writes inputs, runs command, reads outputs.
    # To test 'read outputs', we need the file to exist.

    # We can mock `_write_inputs` and `_parse_output` to test `run` logic specifically
    # OR we can let it try to write and fail parsing if file missing.

    # Easier: Mock the `_parse_output` method to return a structure.
    with patch.object(LammpsRunner, "_parse_output") as mock_parse:
        mock_parse.return_value = minimal_structure

        params = MDParams(temperature=300, n_steps=100)
        result = lammps_runner.run(minimal_structure, params, work_dir=tmp_path)

        assert result.status == JobStatus.COMPLETED
        assert result.final_structure == minimal_structure

        # Verify inputs written
        assert (tmp_path / "in.lammps").exists()
        assert (tmp_path / "data.lammps").exists()

        # Verify command called
        mock_subprocess_run.assert_called_once()

def test_run_failed_subprocess(lammps_runner, minimal_structure, mock_subprocess_run, tmp_path):
    """Test subprocess failure."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Error"
    )

    params = MDParams(temperature=300, n_steps=100)
    result = lammps_runner.run(minimal_structure, params, work_dir=tmp_path)

    assert result.status == JobStatus.FAILED
    assert "Error" in result.log_content

def test_run_timeout(lammps_runner, minimal_structure, mock_subprocess_run, tmp_path):
    """Test timeout."""
    mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="lmp", timeout=10)

    params = MDParams(temperature=300, n_steps=100)
    result = lammps_runner.run(minimal_structure, params, work_dir=tmp_path)

    assert result.status == JobStatus.TIMEOUT

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
