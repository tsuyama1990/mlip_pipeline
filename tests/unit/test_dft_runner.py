import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from ase import Atoms
from mlip_autopipec.dft.runner import QERunner, DFTFatalError
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult

@pytest.fixture
def dft_config():
    return DFTConfig(
        pseudopotential_dir=Path("/tmp"),
        ecutwfc=30.0,
        kspacing=0.05,
        command="pw.x"
    )

@patch("mlip_autopipec.dft.runner.subprocess.run")
@patch("mlip_autopipec.dft.runner.shutil.which")
@patch("mlip_autopipec.dft.runner.InputGenerator")
@patch("mlip_autopipec.dft.runner.QEOutputParser")
def test_runner_success(mock_parser_cls, mock_input_gen, mock_which, mock_run, dft_config):
    mock_which.return_value = "/usr/bin/pw.x"

    # Mock Run: simulate success (usually writes JOB DONE but parser mock ignores it)
    mock_run.return_value.returncode = 0
    mock_run.return_value.stderr = ""

    # Mock Input Generation
    mock_input_gen.create_input_string.return_value = "mock input content"

    # Mock Parser
    mock_parser_instance = mock_parser_cls.return_value
    mock_parser_instance.parse.return_value = DFTResult(
        uid="test", energy=-10.0, forces=[[0,0,0]], stress=[[0,0,0], [0,0,0], [0,0,0]],
        succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
    )

    runner = QERunner(dft_config)
    atoms = Atoms('H', cell=[5,5,5], pbc=True)
    result = runner.run(atoms, uid="test_uid")

    assert result.succeeded
    assert result.energy == -10.0
    mock_run.assert_called()

@patch("mlip_autopipec.dft.runner.subprocess.run")
@patch("mlip_autopipec.dft.runner.shutil.which")
@patch("mlip_autopipec.dft.runner.InputGenerator")
@patch("mlip_autopipec.dft.runner.QEOutputParser")
@patch("mlip_autopipec.dft.runner.RecoveryHandler")
def test_runner_recovery(mock_recovery, mock_parser_cls, mock_input_gen, mock_which, mock_run, dft_config):
    mock_which.return_value = "/usr/bin/pw.x"

    mock_input_gen.create_input_string.return_value = "mock input content"

    # Side effects to simulate writing to STDOUT (file)
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            if kwargs.get('stdout'):
                kwargs['stdout'].write("convergence NOT achieved\n")
                kwargs['stdout'].flush()
            return MagicMock(returncode=1, stderr="")
        else:
            if kwargs.get('stdout'):
                kwargs['stdout'].write("JOB DONE\n")
                kwargs['stdout'].flush()
            return MagicMock(returncode=0, stderr="")

    mock_run.side_effect = side_effect

    # Parser fail on first call (raises Exception), succeed on second
    mock_parser_instance = mock_parser_cls.return_value
    mock_parser_instance.parse.side_effect = [
        Exception("Parse Error"),
        DFTResult(
            uid="test", energy=-10.0, forces=[[0,0,0]], stress=[[0,0,0], [0,0,0], [0,0,0]],
            succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.3
        )
    ]

    # Recovery Handler suggests new params
    from mlip_autopipec.data_models.dft_models import DFTErrorType

    # We verify that analyze is called with correct STDOUT
    def side_effect_analyze(stdout, stderr):
        if "convergence NOT achieved" in stdout:
            return DFTErrorType.CONVERGENCE_FAIL
        return DFTErrorType.NONE

    mock_recovery.analyze.side_effect = side_effect_analyze
    mock_recovery.get_strategy.return_value = {"mixing_beta": 0.3}

    runner = QERunner(dft_config)
    atoms = Atoms('H', cell=[5,5,5], pbc=True)
    result = runner.run(atoms, uid="test_uid")

    assert result.succeeded
    assert mock_run.call_count == 2
    mock_recovery.get_strategy.assert_called()

    # Verify analyze was called with proper text in first arg (stdout)
    args, _ = mock_recovery.analyze.call_args_list[0]
    assert "convergence NOT achieved" in args[0]
    assert args[1] == "" # stderr should be empty

@patch("mlip_autopipec.dft.runner.shutil.which")
def test_runner_missing_executable(mock_which, dft_config):
    mock_which.return_value = None
    runner = QERunner(dft_config)
    with pytest.raises(DFTFatalError, match="not found"):
        runner.run(Atoms('H'))
