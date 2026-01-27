from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.runner import DFTFatalError, DFTRetriableError, QERunner


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
        recoverable=True,
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


def test_validate_command_unsafe(mock_config: DFTConfig) -> None:
    runner = QERunner(mock_config)
    with pytest.raises(DFTFatalError, match="unsafe shell characters"):
        runner._validate_command("pw.x; rm -rf /")

    with pytest.raises(DFTFatalError, match="unsafe shell characters"):
        runner._validate_command("pw.x | grep foo")


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
        uid="test",
        energy=-10.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True,
        converged=True,
        wall_time=1.0,
        parameters={},
    )
    mock_parser_cls.return_value.parse.return_value = mock_result

    runner = QERunner(mock_config, parser_class=mock_parser_cls)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[5, 5, 5])

    result = runner.run(atoms)

    assert result.succeeded
    assert result.energy == -10.0
    mock_run.assert_called()
    mock_parser_cls.assert_called()


@patch("shutil.which")
@patch("mlip_autopipec.dft.runner.subprocess.run")
def test_run_retry_recovery(
    mock_run: MagicMock, mock_which: MagicMock, mock_config: DFTConfig
) -> None:
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
        uid="test",
        energy=-10.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True,
        converged=True,
        wall_time=1.0,
        parameters={},
    )
    mock_parser_cls.return_value.parse.return_value = mock_result

    runner = QERunner(mock_config, parser_class=mock_parser_cls)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[5, 5, 5])

    result = runner.run(atoms)

    assert result.succeeded
    assert mock_run.call_count == 2


@patch("shutil.which")
@patch("mlip_autopipec.dft.runner.subprocess.run")
def test_run_os_error_retry(
    mock_run: MagicMock, mock_which: MagicMock, mock_config: DFTConfig
) -> None:
    mock_which.return_value = "/bin/pw.x"

    # OSError first, then success
    mock_run.side_effect = [OSError("Resource busy"), MagicMock(returncode=0, stderr="")]

    mock_parser_cls = MagicMock()
    mock_result = DFTResult(
        uid="test",
        energy=-10.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True,
        converged=True,
        wall_time=1.0,
        parameters={},
    )
    mock_parser_cls.return_value.parse.return_value = mock_result

    runner = QERunner(mock_config, parser_class=mock_parser_cls)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[5, 5, 5])

    result = runner.run(atoms)

    assert result.succeeded
    assert mock_run.call_count == 2


def test_stage_pseudos(mock_config: DFTConfig, tmp_path: Path) -> None:
    from mlip_autopipec.dft.constants import SSSP_EFFICIENCY_1_1

    # Update constants for test if needed or use real ones.
    # We'll rely on real constant for "Al" if it exists, otherwise we patch.
    # Al is usually in SSSP.

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = QERunner(mock_config)
    atoms = Atoms("Al", positions=[[0, 0, 0]])

    # We need to make sure SSSP_EFFICIENCY_1_1 has Al mapping to "Al.UPF" or whatever our mock config has.
    # Our mock config has "Al.UPF".
    # Let's patch SSSP_EFFICIENCY_1_1
    with patch.dict(SSSP_EFFICIENCY_1_1, {"Al": "Al.UPF"}):
        runner._stage_pseudos(work_dir, atoms)

    assert (work_dir / "Al.UPF").exists()
    assert (work_dir / "Al.UPF").is_symlink()


@patch("shutil.which")
def test_run_batch(mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = "/bin/pw.x"
    runner = QERunner(mock_config)
    runner.run = MagicMock(return_value=DFTResult(
        uid="1", energy=-10.0, forces=[], stress=[], succeeded=True, converged=True, wall_time=1.0, parameters={}
    ))

    atoms_list = [Atoms("Al", positions=[[0, 0, 0]]), Atoms("Al", positions=[[1, 1, 1]])]
    results = list(runner.run_batch(atoms_list))

    assert len(results) == 2
    assert runner.run.call_count == 2


@patch("shutil.which")
def test_input_generation_failure(mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = "/bin/pw.x"
    runner = QERunner(mock_config)

    # Mock InputGenerator to fail
    with patch("mlip_autopipec.dft.runner.InputGenerator.create_input_string", side_effect=Exception("Gen Fail")):
        atoms = Atoms("Al", positions=[[0, 0, 0]])
        result = runner.run(atoms)

    assert result.succeeded is False
    assert "Input generation failed" in result.error_message


@patch("shutil.which")
@patch("mlip_autopipec.dft.runner.subprocess.run")
def test_run_timeout(mock_run: MagicMock, mock_which: MagicMock, mock_config: DFTConfig) -> None:
    mock_which.return_value = "/bin/pw.x"
    import subprocess
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="pw.x", timeout=1.0)

    runner = QERunner(mock_config)
    atoms = Atoms("Al", positions=[[0, 0, 0]])
    result = runner.run(atoms)

    assert result.succeeded is False
    # Depending on implementation, it might retry?
    # QERunner implementation catches TimeoutExpired and sets returncode=-1, then continues to Recovery.
    # Recovery might not handle Timeout, or might retry if configured.
    # In QERunner:
    # except subprocess.TimeoutExpired:
    #    returncode = -1
    #    stderr_content = "Timeout Expired"
    # RecoveryHandler.analyze(...) -> ErrorType.NONE for "Timeout Expired" usually unless matched.
    # If NONE and returncode != 0 -> "Process exited with -1 ... " -> break.

    assert "Process exited with -1" in result.error_message or "Timeout" in result.error_message
