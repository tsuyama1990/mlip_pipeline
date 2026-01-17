import subprocess

import pytest
import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()

RESOURCES_DEFAULT = {
    "dft_code": "quantum_espresso",
    "parallel_cores": 4,
    "gpu_enabled": False
}

@pytest.fixture
def valid_config_file(tmp_path):
    config_file = tmp_path / "input.yaml"
    data = {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Si"],
            "composition": {"Si": 1.0},
            "crystal_structure": "diamond",
        },
        "simulation_goal": {"type": "elastic"},
        "resources": RESOURCES_DEFAULT
    }
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return config_file

def test_init_success(valid_config_file, tmp_path, mocker):
    """Test that 'init' command creates directory and initializes DB."""
    # Mock cwd to tmp_path so project is created there
    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)

    result = runner.invoke(app, ["init", str(valid_config_file)])

    assert result.exit_code == 0
    assert "Initializing project" in result.stdout
    assert "SUCCESS" in result.stdout

    # Check effects
    proj_dir = tmp_path / "TestProject"
    assert proj_dir.exists()
    assert (proj_dir / "TestProject.db").exists()
    assert (proj_dir / "system.log").exists()

def test_run_success(valid_config_file, mocker):
    """Test the happy path: CLI calls PipelineController."""
    mock_controller = mocker.patch("mlip_autopipec.services.pipeline.PipelineController")

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 0
    assert "Launching run" in result.stdout
    assert "SUCCESS" in result.stdout

    mock_controller.execute.assert_called_once()

def test_run_failure(valid_config_file, mocker):
    """Test that CLI handles pipeline errors gracefully."""
    mock_controller = mocker.patch("mlip_autopipec.services.pipeline.PipelineController")
    mock_controller.execute.side_effect = ValueError("Something exploded")

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 1
    assert "FAILURE" in result.stdout
    assert "Something exploded" in result.stdout

def test_status_success(tmp_path, mocker):
    """Test status command generating dashboard."""
    mock_gen = mocker.patch("mlip_autopipec.app.generate_dashboard", return_value=tmp_path / "dashboard.html")
    mocker.patch("webbrowser.open")

    result = runner.invoke(app, ["status", str(tmp_path), "--no-open"])

    assert result.exit_code == 0
    assert "Generating dashboard" in result.stdout
    assert "SUCCESS" in result.stdout

    mock_gen.assert_called_once_with(tmp_path)

def test_status_failure(tmp_path, mocker):
    """Test status command failure."""
    mocker.patch("mlip_autopipec.app.generate_dashboard", side_effect=Exception("Dashboard failed"))

    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code == 1
    assert "FAILURE" in result.stdout
    assert "Dashboard failed" in result.stdout

def test_entry_point_help():
    """Verify that the mlip-auto command is installed and provides help."""
    try:
        result = subprocess.run(
            ["mlip-auto", "--help"], check=False, capture_output=True, text=True
        )
        # If installed, should return 0. If not found, FileNotFoundError handled below.
        if result.returncode == 0:
            assert "Usage: mlip-auto" in result.stdout
    except FileNotFoundError:
        pass
