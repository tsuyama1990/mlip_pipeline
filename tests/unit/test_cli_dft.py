from unittest.mock import patch

from ase import Atoms
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult

runner = CliRunner()

@patch("mlip_autopipec.dft.runner.QERunner")
@patch("mlip_autopipec.config.loaders.yaml_loader.load_config")
@patch("ase.io.read")
def test_run_dft_success(mock_read, mock_load, MockRunner, tmp_path):
    # Setup
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    struct_file = tmp_path / "struct.xyz"
    struct_file.touch()

    # Return real DFTConfig
    mock_load.return_value = DFTConfig(
        command="pw.x",
        pseudopotential_dir=tmp_path,
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.1
    )

    mock_read.return_value = [Atoms("H")]

    mock_runner_instance = MockRunner.return_value
    mock_runner_instance.run.return_value = DFTResult(
        uid="test", energy=-10.0, forces=[[0.0, 0.0, 0.0]], stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True, converged=True, wall_time=1.0, parameters={}
    )

    result = runner.invoke(app, ["run-dft", "--config", str(config_file), "--structure", str(struct_file)])

    print(result.stdout)
    assert result.exit_code == 0
    assert "DFT Calculation Successful" in result.stdout
    assert "Energy: -10.0 eV" in result.stdout

@patch("mlip_autopipec.dft.runner.QERunner")
@patch("mlip_autopipec.config.loaders.yaml_loader.load_config")
@patch("ase.io.read")
def test_run_dft_failure(mock_read, mock_load, MockRunner, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    struct_file = tmp_path / "struct.xyz"
    struct_file.touch()

    mock_load.return_value = DFTConfig(
        command="pw.x",
        pseudopotential_dir=tmp_path,
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.1
    )
    mock_read.return_value = [Atoms("H")]

    mock_runner_instance = MockRunner.return_value
    mock_runner_instance.run.return_value = DFTResult(
        uid="test", energy=0.0, forces=[], stress=[], succeeded=False,
        converged=False, error_message="SCF failed",
        wall_time=0.0, parameters={}
    )

    result = runner.invoke(app, ["run-dft", "--config", str(config_file), "--structure", str(struct_file)])

    assert result.exit_code == 1
    assert "DFT Calculation Failed" in result.stdout
    assert "SCF failed" in result.stdout

@patch("mlip_autopipec.config.loaders.yaml_loader.load_config")
@patch("ase.io.read")
def test_run_dft_invalid_atoms(mock_read, mock_load, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    struct_file = tmp_path / "struct.xyz"
    struct_file.touch()

    mock_load.return_value = DFTConfig(
        command="pw.x",
        pseudopotential_dir=tmp_path,
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.1
    )
    mock_read.return_value = "Not an Atoms object"

    result = runner.invoke(app, ["run-dft", "--config", str(config_file), "--structure", str(struct_file)])

    assert result.exit_code == 1
    assert "Invalid structure file" in result.stdout
