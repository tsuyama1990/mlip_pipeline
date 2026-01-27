import pytest
from unittest.mock import MagicMock, patch
from ase import Atoms
from pathlib import Path
import os

from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult

@pytest.fixture
def mock_dft_config(tmp_path):
    p_dir = tmp_path / "pseudos"
    p_dir.mkdir()
    (p_dir / "Si.upf").touch()

    return DFTConfig(
        command="mpirun -np 4 pw.x",
        pseudopotential_dir=p_dir,
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.1
    )

def test_runner_initialization(mock_dft_config, tmp_path):
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)
    assert runner.config == mock_dft_config
    assert runner.work_dir == tmp_path

@patch("shutil.which")
@patch("subprocess.run")
def test_compute_success(mock_run, mock_which, mock_dft_config, tmp_path):
    mock_which.return_value = "/bin/mpirun"
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)
    atoms = Atoms("Si2", positions=[[0,0,0], [1.1,1.1,1.1]], cell=[5,5,5])

    mock_run.return_value = MagicMock(returncode=0, stdout="JOB DONE", stderr="")

    with patch.object(runner, '_write_input') as mock_write, \
         patch.object(runner, '_parse_output') as mock_parse:

        mock_parse.return_value = DFTResult(
            energy=-100.0,
            forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            converged=True
        )

        # Create dummy input file since _write_input is mocked and _run_command expects it
        (runner.work_dir / "pw.in").touch()

        result = runner.run(atoms)

        assert result.energy == -100.0
        assert result.converged is True
        mock_write.assert_called_once()
        mock_run.assert_called_once()

@patch("shutil.which")
@patch("subprocess.run")
def test_compute_failure(mock_run, mock_which, mock_dft_config, tmp_path):
    mock_which.return_value = "/bin/mpirun"
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)
    atoms = Atoms("Si")

    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

    result = runner.run(atoms)
    assert result.converged is False
    assert result.error_message is not None

def test_write_input(mock_dft_config, tmp_path):
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)
    atoms = Atoms("Si", positions=[[0,0,0]], cell=[4,4,4])

    input_file = tmp_path / "pw.in"
    runner._write_input(atoms, input_file)

    assert input_file.exists()
    content = input_file.read_text()

    assert "tprnfor" in content or "tprnfor=.true." in content.lower()
    assert "tstress" in content or "tstress=.true." in content.lower()
    assert "Si.upf" in content
    assert "K_POINTS" in content

@patch("mlip_autopipec.dft.runner.read")
def test_parse_output_failure(mock_read, mock_dft_config, tmp_path):
    runner = QERunner(config=mock_dft_config, work_dir=tmp_path)
    output_file = tmp_path / "pw.out"
    output_file.touch()

    mock_read.side_effect = Exception("Parsing error")

    result = runner._parse_output(output_file)
    assert result.converged is False
    assert "Parsing error" in result.error_message
