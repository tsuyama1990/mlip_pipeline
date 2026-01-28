from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.dft_models import DFTResult

runner = CliRunner()


@pytest.fixture
def valid_dft_config_content(tmp_path):
    # Create valid pseudo dir with UPF
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Si.UPF").touch()

    return f"""
target_system:
  name: Test System
  elements: [Si]
  composition: {{Si: 1.0}}
dft:
  pseudopotential_dir: {pseudo_dir}
  command: pw.x
"""


@patch("ase.io.read")
@patch("mlip_autopipec.modules.cli_handlers.handlers.QERunner")
def test_run_dft_success_real_load(mock_qerunner, mock_read, tmp_path, valid_dft_config_content):
    config_file = tmp_path / "config.yaml"
    structure_file = tmp_path / "struct.cif"
    structure_file.touch()

    with config_file.open("w") as f:
        f.write(valid_dft_config_content)

    from ase import Atoms
    mock_read.return_value = Atoms("Si", positions=[[0, 0, 0]])

    # Mock Runner Instance
    mock_instance = mock_qerunner.return_value
    mock_instance.run.return_value = DFTResult(
        uid="1",
        energy=-10.0,
        forces=[[0.0, 0.0, 0.0]], # Shape (1,3)
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], # 3x3
        succeeded=True,
        converged=True,
        wall_time=1.0,
        parameters={},
    )

    result = runner.invoke(
        app, ["run-dft", "--config", str(config_file), "--structure", str(structure_file)]
    )

    assert result.exit_code == 0
    assert "DFT Calculation Successful" in result.stdout
    assert "Energy: -10.0 eV" in result.stdout


def test_run_dft_missing_files(tmp_path):
    result = runner.invoke(
        app, ["run-dft", "--config", "missing.yaml", "--structure", "struct.cif"]
    )
    assert result.exit_code != 0


@patch("ase.io.read")
def test_run_dft_invalid_structure(mock_read, tmp_path, valid_dft_config_content):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        f.write(valid_dft_config_content)

    structure_file = tmp_path / "struct.cif"
    structure_file.touch()

    # Return something not Atoms
    mock_read.return_value = "NotAtoms"

    result = runner.invoke(
        app, ["run-dft", "--config", str(config_file), "--structure", str(structure_file)]
    )
    assert result.exit_code == 1
    assert "Invalid structure" in result.stdout


@patch("mlip_autopipec.modules.cli_handlers.handlers.QERunner")
@patch("ase.io.read")
def test_run_dft_calculation_failure(mock_read, mock_qerunner, tmp_path, valid_dft_config_content):
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        f.write(valid_dft_config_content)

    structure_file = tmp_path / "struct.cif"
    structure_file.touch()

    from ase import Atoms
    mock_read.return_value = Atoms("Si", positions=[[0, 0, 0]])

    mock_instance = mock_qerunner.return_value
    mock_instance.run.return_value = DFTResult(
        uid="1",
        energy=0.0,
        forces=[],
        stress=[],
        succeeded=False,
        converged=False,
        error_message="SCF not converged",
        wall_time=1.0,
        parameters={},
    )

    result = runner.invoke(
        app, ["run-dft", "--config", str(config_file), "--structure", str(structure_file)]
    )

    assert result.exit_code == 1
    assert "DFT Calculation Failed" in result.stdout
    assert "SCF not converged" in result.stdout
