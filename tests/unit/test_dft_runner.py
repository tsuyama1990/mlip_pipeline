from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.runner import DFTFatalError, QERunner


@pytest.fixture
def dft_config(tmp_path):
    (tmp_path / "pseudos").mkdir()
    return DFTConfig(
        pseudopotential_dir=tmp_path / "pseudos",
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

    # Verify shell=False usage
    mock_run.assert_called()
    call_args = mock_run.call_args
    assert call_args.kwargs['shell'] is False
    assert isinstance(call_args.args[0], list)
    assert call_args.args[0] == ['pw.x', '-in', 'pw.in']

@patch("mlip_autopipec.dft.runner.subprocess.run")
@patch("mlip_autopipec.dft.runner.shutil.which")
@patch("mlip_autopipec.dft.runner.InputGenerator")
@patch("mlip_autopipec.dft.runner.QEOutputParser")
@patch("mlip_autopipec.dft.runner.RecoveryHandler")
def test_runner_recovery(mock_recovery, mock_parser_cls, mock_input_gen, mock_which, mock_run, dft_config):
    mock_which.return_value = "/usr/bin/pw.x"

    mock_input_gen.create_input_string.return_value = "mock input content"

    # Side effects to simulate writing to output file
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        # We need to simulate writing to the stdout file handle passed in kwargs
        stdout_f = kwargs['stdout']

        if call_count == 1:
            stdout_f.write("convergence NOT achieved\n")
            stdout_f.flush()
            return MagicMock(returncode=1, stderr="")
        stdout_f.write("JOB DONE\n")
        stdout_f.flush()
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

@patch("mlip_autopipec.dft.runner.shutil.which")
def test_runner_missing_executable(mock_which, dft_config):
    mock_which.return_value = None
    runner = QERunner(dft_config)
    with pytest.raises(DFTFatalError, match="not found"):
        runner.run(Atoms('H'))

@patch("mlip_autopipec.dft.runner.shutil.which")
def test_runner_command_validation(mock_which, dft_config):
    runner = QERunner(dft_config)
    mock_which.return_value = "/bin/pw.x"

    # Valid
    parts = runner._validate_command("pw.x")
    assert parts == ["pw.x"]

    parts = runner._validate_command("mpirun -np 4 pw.x")
    assert parts == ["mpirun", "-np", "4", "pw.x"]

    # Empty
    with pytest.raises(DFTFatalError):
        runner._validate_command("")
