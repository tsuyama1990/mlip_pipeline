import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from ase.build import bulk
import subprocess

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTError
from mlip_autopipec.domain_models.job import JobStatus, JobResult
from mlip_autopipec.physics.dft.qe_runner import QERunner

SUCCESS_OUTPUT = """
     Program PWSCF v.7.0 starts on  1Jan2025 at 00:00:00
     !    total energy              =     -10.0 Ry
     JOB DONE.
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.0    0.0    0.0
"""

SCF_FAIL_OUTPUT = """
     convergence not achieved
"""

@pytest.fixture
def structure():
    atoms = bulk("Si")
    return Structure.from_ase(atoms)

@pytest.fixture
def config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.upf")},
        ecutwfc=30.0,
        kspacing=0.04,
        mixing_beta=0.7
    )

@pytest.fixture
def runner(tmp_path):
    return QERunner(base_work_dir=tmp_path)

def test_qe_runner_success(runner, structure, config):
    with patch.object(runner, '_execute', return_value=(SUCCESS_OUTPUT, "")) as mock_exec:
        with patch.object(runner, '_setup_pseudopotentials'):
            result = runner.run(structure, config)
            assert result.status == JobStatus.COMPLETED
            assert result.energy != 0.0
            assert mock_exec.call_count == 1

def test_qe_runner_recovery(runner, structure, config):
    with patch.object(runner, '_execute', side_effect=[(SCF_FAIL_OUTPUT, ""), (SUCCESS_OUTPUT, "")]) as mock_exec:
        with patch.object(runner, '_setup_pseudopotentials'):
            result = runner.run(structure, config)
            assert result.status == JobStatus.COMPLETED
            assert mock_exec.call_count == 2
            args2, _ = mock_exec.call_args_list[1]
            mod_config = args2[1]
            assert mod_config.mixing_beta == 0.3

def test_qe_runner_max_retries_fail(runner, structure, config):
    runner.max_retries = 2
    with patch.object(runner, '_execute', return_value=(SCF_FAIL_OUTPUT, "")) as mock_exec:
        with patch.object(runner, '_setup_pseudopotentials'):
            result = runner.run(structure, config)
            assert result.status == JobStatus.FAILED
            assert mock_exec.call_count == 2

def test_max_retry_respect(runner, structure, config):
    # Explicitly test max_retries=3 means 3 calls
    runner.max_retries = 3
    with patch.object(runner, '_execute', return_value=(SCF_FAIL_OUTPUT, "")) as mock_exec:
        with patch.object(runner, '_setup_pseudopotentials'):
            result = runner.run(structure, config)
            assert result.status == JobStatus.FAILED
            assert mock_exec.call_count == 3

# Coverage tests

def test_execute_calls_subprocess(runner, config):
    # Test _execute directly
    work_dir = Path("/tmp/work")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="out", stderr="err", returncode=0)

        stdout, stderr = runner._execute(work_dir, config)

        assert stdout == "out"
        assert stderr == "err"

        # Check command
        # default mpi="mpirun -np 1", command="pw.x"
        expected_cmd = ["mpirun", "-np", "1", "pw.x", "-in", "pw.in"]
        mock_run.assert_called_with(
            expected_cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=config.timeout,
            check=False
        )

def test_execute_timeout(runner, config):
    work_dir = Path("/tmp/work")
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pw.x", timeout=10)):
        with pytest.raises(subprocess.TimeoutExpired):
             runner._execute(work_dir, config)

def test_setup_pseudopotentials(runner, config, tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Create dummy source file
    src_file = tmp_path / "Si.upf"
    src_file.write_text("dummy")

    # Update config to point to real file
    config.pseudopotentials["Si"] = src_file

    runner._setup_pseudopotentials(work_dir, config)

    dst_file = work_dir / "Si.upf"
    assert dst_file.exists()
    assert dst_file.is_symlink()
    assert dst_file.read_text() == "dummy"

def test_run_timeout_exception(runner, structure, config):
    # Simulate Timeout in _execute
    with patch.object(runner, '_execute', side_effect=subprocess.TimeoutExpired(cmd="pw.x", timeout=10)):
        with patch.object(runner, '_setup_pseudopotentials'):
            runner.max_retries = 1
            result = runner.run(structure, config)
            assert result.status == JobStatus.FAILED
            assert "Timeout Expired" in result.log_content

def test_run_setup_failure(runner, structure, config):
    with patch.object(runner, '_setup_pseudopotentials', side_effect=Exception("Setup failed")):
         result = runner.run(structure, config)
         assert result.status == JobStatus.FAILED
         assert "Setup failed" in result.log_content
