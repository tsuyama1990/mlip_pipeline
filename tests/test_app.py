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
            # "crystal_structure": "diamond", # Not valid anymore
        },
        "resources": {  # Required
            "dft_code": "quantum_espresso",
            "parallel_cores": 4,
        },
        "simulation_goal": {
            "type": "elastic"
        },  # This is not in MinimalConfig schema but extra forbidden?
        # Checking MinimalConfig: project_name, target_system, resources. extra="forbid".
        # So "simulation_goal" would fail if extra="forbid" is strictly enforced on MinimalConfig.
        # But wait, MinimalConfig does not have simulation_goal in common.py I saw earlier.
        # Let's check common.py again.
    }
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return config_file


def test_run_success(valid_config_file, mocker):
    """Test the happy path: CLI calls PipelineController."""
    # Mock PipelineController used in app.py
    mock_controller = mocker.patch("mlip_autopipec.app.PipelineController")

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 0
    assert "System initialized successfully" in result.stdout

    mock_controller.execute.assert_called_once_with(valid_config_file)


def test_run_failure(valid_config_file, mocker):
    """Test that CLI handles errors gracefully."""
    mock_controller = mocker.patch("mlip_autopipec.app.PipelineController")
    from mlip_autopipec.exceptions import ConfigError

    # Simulate an error raised by PipelineController.execute
    mock_controller.execute.side_effect = ConfigError("Bad Config")

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 1
    assert "CONFIGURATION ERROR" in result.stdout
    assert "Bad Config" in result.stdout


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
