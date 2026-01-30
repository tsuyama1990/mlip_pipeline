import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from mlip_autopipec.domain_models.config import LammpsConfig, MDParams
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner
import numpy as np
import subprocess

@pytest.fixture
def mock_structure():
    return Structure(
        symbols=["Si"],
        positions=np.array([[0,0,0]]),
        cell=np.eye(3),
        pbc=(True, True, True)
    )

@pytest.fixture
def md_params():
    return MDParams(temperature=300, timestep=0.001, n_steps=100)

@pytest.fixture
def lammps_config():
    return LammpsConfig(command="lmp_serial", cores=1, timeout=10)

def test_lammps_runner_init(lammps_config):
    runner = LammpsRunner(lammps_config)
    assert runner.config == lammps_config

@patch("subprocess.run")
@patch("mlip_autopipec.physics.dynamics.lammps.LammpsRunner._write_inputs")
@patch("mlip_autopipec.physics.dynamics.lammps.LammpsRunner._parse_output")
def test_run_success(mock_parse, mock_write, mock_sub, lammps_config, mock_structure, md_params, tmp_path):
    runner = LammpsRunner(lammps_config)

    # Mock subprocess success
    mock_sub.return_value.returncode = 0
    mock_sub.return_value.stdout = "Simulation done"

    # Mock parse output
    mock_parse.return_value = (mock_structure, Path("dump.lammpstrj"))

    result = runner.run(mock_structure, md_params, work_dir=tmp_path)

    assert result.status == JobStatus.COMPLETED
    assert result.final_structure == mock_structure

@patch("subprocess.run")
def test_run_failure(mock_sub, lammps_config, mock_structure, md_params, tmp_path):
    runner = LammpsRunner(lammps_config)

    mock_sub.return_value.returncode = 1
    mock_sub.return_value.stderr = "Error"
    mock_sub.return_value.stdout = ""

    # We might need to mock _write_inputs to avoid file IO or errors
    with patch("mlip_autopipec.physics.dynamics.lammps.LammpsRunner._write_inputs"):
        result = runner.run(mock_structure, md_params, work_dir=tmp_path)

    assert result.status == JobStatus.FAILED

def test_write_inputs(lammps_config, mock_structure, md_params, tmp_path):
    runner = LammpsRunner(lammps_config)
    runner._write_inputs(tmp_path, mock_structure, md_params)

    assert (tmp_path / "data.lammps").exists()
    assert (tmp_path / "in.lammps").exists()
    content = (tmp_path / "in.lammps").read_text()
    assert f"fix             1 all nvt temp {md_params.temperature}" in content
    assert "pair_style      hybrid/overlay lj/cut 2.5 zbl 4.0 5.0" in content
    assert "pair_coeff      * * lj/cut 1.0 1.0" in content
    assert "pair_coeff      1 1 zbl 14 14" in content  # Si is 14

def test_parse_output(lammps_config, tmp_path):
    runner = LammpsRunner(lammps_config)

    # Create dummy dump file
    dump_file = tmp_path / "dump.lammpstrj"
    # ASE dump format needs specific header
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
"""
    dump_file.write_text(dump_content)

    struct, path = runner._parse_output(tmp_path)
    assert isinstance(struct, Structure)
    # With 1 atom at 0,0,0
    assert len(struct.positions) == 1
    assert path == dump_file

def test_run_temp_dir(lammps_config, mock_structure, md_params):
    runner = LammpsRunner(lammps_config)

    with patch.object(runner, "_execute_in_dir") as mock_exec:
        mock_exec.return_value = MagicMock()
        runner.run(mock_structure, md_params, work_dir=None)

        # Check that it was called with a path
        args, _ = mock_exec.call_args
        assert isinstance(args[0], Path)

def test_run_input_gen_failure(lammps_config, mock_structure, md_params, tmp_path):
    runner = LammpsRunner(lammps_config)
    with patch("mlip_autopipec.physics.dynamics.lammps.LammpsRunner._write_inputs", side_effect=Exception("Write failed")):
         result = runner.run(mock_structure, md_params, work_dir=tmp_path)
    assert result.status == JobStatus.FAILED
    assert "Input generation failed" in result.log_content

def test_run_parse_failure(lammps_config, mock_structure, md_params, tmp_path):
    runner = LammpsRunner(lammps_config)
    with patch("subprocess.run") as mock_sub, \
         patch("mlip_autopipec.physics.dynamics.lammps.LammpsRunner._write_inputs"), \
         patch("mlip_autopipec.physics.dynamics.lammps.LammpsRunner._parse_output", side_effect=Exception("Parse failed")):

         mock_sub.return_value.returncode = 0
         mock_sub.return_value.stdout = "Done"

         result = runner.run(mock_structure, md_params, work_dir=tmp_path)

    assert result.status == JobStatus.FAILED
    assert "Parsing failed" in result.log_content

def test_run_lammps_timeout(lammps_config, tmp_path):
    runner = LammpsRunner(lammps_config)
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["cmd"], 10)):
        out, err, code, dur = runner._run_lammps(tmp_path)
    assert code == 124
    assert err == "Timeout"

def test_run_lammps_exec_error(lammps_config, tmp_path):
    runner = LammpsRunner(lammps_config)
    with patch("subprocess.run", side_effect=Exception("Boom")):
        out, err, code, dur = runner._run_lammps(tmp_path)
    assert code == 1
    assert "Execution error" in err

def test_parse_output_missing(lammps_config, tmp_path):
    runner = LammpsRunner(lammps_config)
    with pytest.raises(FileNotFoundError):
        runner._parse_output(tmp_path)
