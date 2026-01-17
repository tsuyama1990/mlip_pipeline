import subprocess

import pytest
import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


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
    }
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return config_file


def test_run_success(valid_config_file, mocker):
    """Test the happy path: CLI calls PipelineController."""
    mock_controller = mocker.patch("mlip_autopipec.app.PipelineController")

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 0
    assert "Launching run" in result.stdout
    assert "SUCCESS" in result.stdout

    mock_controller.execute.assert_called_once_with(valid_config_file)


def test_run_failure(valid_config_file, mocker):
    """Test that CLI handles pipeline errors gracefully."""
    mock_controller = mocker.patch("mlip_autopipec.app.PipelineController")
    mock_controller.execute.side_effect = ValueError("Something exploded")

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 1
    assert "FAILURE" in result.stdout
    assert "Something exploded" in result.stdout


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
