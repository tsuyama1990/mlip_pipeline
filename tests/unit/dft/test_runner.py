from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.runner import DFTFatalError, QERunner


@pytest.fixture
def mock_config(tmp_path: Path) -> DFTConfig:
    # create fake UPF
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Al.UPF").touch()

    return DFTConfig(
        pseudopotential_dir=pseudo_dir,
        ecutwfc=30.0,
        kspacing=0.05,
        command="pw.x",
        recoverable=True
    )

@patch("shutil.which")
def test_validate_command_success(mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = "/usr/bin/pw.x"
    runner = QERunner(mock_config)
    parts = runner._validate_command("pw.x")
    assert parts == ["pw.x"]

@patch("shutil.which")
def test_validate_command_fail(mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = None
    runner = QERunner(mock_config)
    with pytest.raises(DFTFatalError, match="not found"):
        runner._validate_command("pw.x")

@patch("shutil.which")
@patch("mlip_autopipec.dft.runner.subprocess.run")
def test_run_success(mock_run: MagicMock, mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = "/bin/pw.x"

    # Mock subprocess success
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stderr = ""
    mock_run.return_value = mock_process

    # Mock Parser via Dependency Injection
    mock_parser_cls = MagicMock()
    mock_result = DFTResult(
        uid="test", energy=-10.0, forces=[[0.0, 0.0, 0.0]], stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True, converged=True, wall_time=1.0, parameters={}
    )
    mock_parser_cls.return_value.parse.return_value = mock_result

    runner = QERunner(mock_config, parser_class=mock_parser_cls)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[5,5,5])

    result = runner.run(atoms)

    assert result.succeeded
    assert result.energy == -10.0
    mock_run.assert_called()
    mock_parser_cls.assert_called()

@patch("shutil.which")
@patch("mlip_autopipec.dft.runner.subprocess.run")
def test_run_retry_recovery(mock_run: MagicMock, mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = "/bin/pw.x"

    # Fail first time (Convergence), Succeed second time
    proc_fail = MagicMock()
    proc_fail.returncode = 1
    proc_fail.stderr = "convergence NOT achieved"

    proc_success = MagicMock()
    proc_success.returncode = 0
    proc_success.stderr = ""

    mock_run.side_effect = [proc_fail, proc_success]

    # Mock Parser to succeed on second call
    mock_parser_cls = MagicMock()
    mock_result = DFTResult(
        uid="test", energy=-10.0, forces=[[0.0, 0.0, 0.0]], stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True, converged=True, wall_time=1.0, parameters={}
    )
    mock_parser_cls.return_value.parse.return_value = mock_result

    runner = QERunner(mock_config, parser_class=mock_parser_cls)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[5,5,5])

    result = runner.run(atoms)

    assert result.succeeded
    assert mock_run.call_count == 2
