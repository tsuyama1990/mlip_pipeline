from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult

runner = CliRunner()

@pytest.fixture
def valid_dft_config_content(tmp_path):
    # Create valid pseudo dir with UPF
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Si.UPF").touch()

    return f"""
dft:
  pseudopotential_dir: {pseudo_dir}
  command: pw.x
"""

@patch("mlip_autopipec.config.loaders.yaml_loader.load_config")
@patch("ase.io.read")
@patch("mlip_autopipec.dft.runner.QERunner")
def test_run_dft_success(mock_qerunner, mock_read, mock_load, tmp_path, valid_dft_config_content):
    # Setup
    config_file = tmp_path / "config.yaml"
    structure_file = tmp_path / "struct.cif"
    structure_file.touch()

    # Write valid config
    with config_file.open("w") as f:
        f.write(valid_dft_config_content)

    # Mock ASE read
    from ase import Atoms
    mock_read.return_value = Atoms("Si", positions=[[0,0,0]])

    # Mock Runner
    mock_instance = mock_qerunner.return_value
    mock_instance.run.return_value = DFTResult(
        uid="1", energy=-10.0, forces=[], stress=[], succeeded=True, converged=True, wall_time=1.0, parameters={}
    )

    result = runner.invoke(app, ["run-dft", "--config", str(config_file), "--structure", str(structure_file)])

    assert result.exit_code == 0
    assert "DFT Calculation Successful" in result.stdout
    assert "Energy: -10.0 eV" in result.stdout


def test_run_dft_missing_files(tmp_path):
    result = runner.invoke(app, ["run-dft", "--config", "missing.yaml", "--structure", "struct.cif"])
    assert result.exit_code == 1
    assert "Config file not found" in result.stdout or "Error" in result.stdout


@patch("ase.io.read")
def test_run_dft_invalid_structure(mock_read, tmp_path, valid_dft_config_content):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        f.write(valid_dft_config_content)

    structure_file = tmp_path / "struct.cif"
    structure_file.touch()

    # Return something not Atoms
    mock_read.return_value = "NotAtoms"

    result = runner.invoke(app, ["run-dft", "--config", str(config_file), "--structure", str(structure_file)])
    assert result.exit_code == 1
    assert "Invalid structure" in result.stdout


@patch("mlip_autopipec.dft.runner.QERunner")
@patch("ase.io.read")
def test_run_dft_calculation_failure(mock_read, mock_qerunner, tmp_path, valid_dft_config_content):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        f.write(valid_dft_config_content)

    structure_file = tmp_path / "struct.cif"
    structure_file.touch()

    from ase import Atoms
    mock_read.return_value = Atoms("Si", positions=[[0,0,0]])

    mock_instance = mock_qerunner.return_value
    mock_instance.run.return_value = DFTResult(
        uid="1", energy=0.0, forces=[], stress=[], succeeded=False, converged=False,
        error_message="SCF not converged", wall_time=1.0, parameters={}
    )

    result = runner.invoke(app, ["run-dft", "--config", str(config_file), "--structure", str(structure_file)])

    assert result.exit_code == 1
    assert "DFT Calculation Failed" in result.stdout
    assert "SCF not converged" in result.stdout
