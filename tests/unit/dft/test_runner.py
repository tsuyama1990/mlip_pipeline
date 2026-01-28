import subprocess
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.dft.runner import DFTFatalError, QERunner


@pytest.fixture
def mock_dft_config(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "H.UPF").touch()
    return DFTConfig(
        command="pw.x",
        pseudopotential_dir=pseudo_dir,
        work_dir=tmp_path / "work",
        timeout=10,
        recoverable=True,
        max_retries=1,
    )


@pytest.fixture
def runner(mock_dft_config):
    return QERunner(config=mock_dft_config)


def test_validate_command_valid(runner):
    with patch("shutil.which", return_value="/usr/bin/pw.x"):
        cmd = runner._validate_command("pw.x -np 4")
        assert cmd == ["pw.x", "-np", "4"]


def test_validate_command_injection(runner):
    with pytest.raises(DFTFatalError, match="unsafe shell characters"):
        runner._validate_command("pw.x; rm -rf /")


def test_validate_command_not_found(runner):
    with patch("shutil.which", return_value=None):
        with pytest.raises(DFTFatalError, match="not found"):
            runner._validate_command("invalid_cmd")


def test_stage_pseudos(runner, mock_dft_config, tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    work_dir = tmp_path / "job_work"
    work_dir.mkdir()

    # Mock SSSP constants
    with patch.dict(
        "mlip_autopipec.dft.constants.SSSP_EFFICIENCY_1_1", {"H": "H.UPF"}, clear=True
    ):
        runner._stage_pseudos(work_dir, atoms)

    assert (work_dir / "H.UPF").exists()
    assert (work_dir / "H.UPF").is_symlink()


@patch("subprocess.run")
def test_execute_subprocess_success(mock_run, runner, tmp_path):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with open(tmp_path / "stdout", "w") as f:
        runner._execute_subprocess_with_retry(["ls"], tmp_path, f, 10.0)
    assert mock_run.call_count == 1


@patch("subprocess.run")
def test_execute_subprocess_retry(mock_run, runner, tmp_path):
    # First fail with OSError, then success
    mock_run.side_effect = [OSError("Resource busy"), subprocess.CompletedProcess([], 0, "", "")]

    with open(tmp_path / "stdout", "w") as f:
        runner._execute_subprocess_with_retry(["ls"], tmp_path, f, 10.0)

    assert mock_run.call_count == 2
