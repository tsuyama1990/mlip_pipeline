from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.runner import DFTFatalError, QERunner


@pytest.fixture
def mock_dft_config(tmp_path):
    p_dir = tmp_path / "pseudos"
    p_dir.mkdir()
    (p_dir / "Si.upf").touch()

    return DFTConfig(
        command="mpirun -np 4 pw.x",
        pseudopotential_dir=p_dir,
        pseudopotentials={"Si": "Si.upf"},
        ecutwfc=30.0,
        kspacing=0.05,
        recoverable=True,
    )


def test_runner_initialization(mock_dft_config, tmp_path):
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)
    assert runner.config == mock_dft_config
    assert runner.work_dir == tmp_path
    assert runner.work_dir.exists()


def test_validate_command_security():
    # Test valid command
    runner = MagicMock(spec=QERunner)
    QERunner._validate_command(runner, "pw.x")

    # Test unsafe characters
    with pytest.raises(DFTFatalError, match="unsafe shell characters"):
        QERunner._validate_command(runner, "pw.x; rm -rf /")

    with pytest.raises(DFTFatalError, match="unsafe shell characters"):
        QERunner._validate_command(runner, "pw.x | bash")


@patch("subprocess.run")
@patch("shutil.which")
def test_run_command_success(mock_which, mock_run, mock_dft_config, tmp_path):
    mock_which.return_value = "/bin/pw.x"
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)

    input_path = tmp_path / "pw.in"
    output_path = tmp_path / "pw.out"
    input_path.touch()

    success, msg = runner._run_command(input_path, output_path)

    assert success is True
    assert msg == ""
    mock_run.assert_called_once()


@patch("subprocess.run")
@patch("shutil.which")
def test_run_command_failure(mock_which, mock_run, mock_dft_config, tmp_path):
    mock_which.return_value = "/bin/pw.x"
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)

    input_path = tmp_path / "pw.in"
    output_path = tmp_path / "pw.out"
    input_path.touch()

    mock_run.side_effect = Exception("Process failed")

    success, msg = runner._run_command(input_path, output_path)

    assert success is False
    assert "Process failed" in msg


def test_write_input(mock_dft_config, tmp_path):
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[4, 4, 4])

    input_file = tmp_path / "pw.in"

    with patch("mlip_autopipec.dft.runner.write") as mock_write:
        runner._write_input(atoms, input_file)
        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["format"] == "espresso-in"
        assert call_kwargs["input_data"]["system"]["ecutwfc"] == 30.0


@patch("mlip_autopipec.dft.runner.QERunner._run_command")
@patch("mlip_autopipec.dft.runner.QERunner._write_input")
@patch("mlip_autopipec.dft.runner.QERunner._parse_output")
@patch("mlip_autopipec.dft.runner.QERunner._stage_pseudos")
def test_run_integration(
    mock_stage, mock_parse, mock_write, mock_run_cmd, mock_dft_config, tmp_path
):
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[4, 4, 4])

    mock_run_cmd.return_value = (True, "")
    mock_parse.return_value = DFTResult(energy=-10.0, forces=[[0.0, 0.0, 0.0]], converged=True)

    result = runner.run(atoms)

    assert result.converged is True
    assert result.energy == -10.0
    mock_write.assert_called()
    mock_run_cmd.assert_called()
    mock_parse.assert_called()
