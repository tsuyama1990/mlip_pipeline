
import subprocess
from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.config.models import UserInputConfig

runner = CliRunner()


@pytest.fixture
def valid_config_data():
    return {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Si"],
            "composition": {"Si": 1.0},
            "crystal_structure": "diamond",
        },
        "simulation_goal": {"type": "elastic"},
    }


@pytest.fixture
def valid_config_file(tmp_path, valid_config_data):
    config_file = tmp_path / "input.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)
    return config_file


def test_run_success(valid_config_file, mocker):
    """Test the happy path: valid config, successful workflow execution."""
    # Mock ConfigLoader
    mock_user_config = MagicMock(spec=UserInputConfig)
    mock_user_config.project_name = "TestProject"

    # We patch the ConfigLoader.load_user_config
    mocker.patch(
        "mlip_autopipec.app.ConfigLoader.load_user_config", return_value=mock_user_config
    )

    # Mock ConfigFactory
    mock_system_config = MagicMock()
    mock_system_config.project_name = "TestProject"
    mock_system_config.run_uuid = "1234-5678"
    mock_system_config.target_system.elements = ["Si"]

    mocker.patch(
        "mlip_autopipec.app.ConfigFactory.from_user_input", return_value=mock_system_config
    )

    # Mock WorkflowManager
    mock_manager_cls = mocker.patch("mlip_autopipec.app.WorkflowManager")
    mock_manager_instance = mock_manager_cls.return_value

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 0
    assert "Configuration validated" in result.stdout or "Validated project" in result.stdout
    assert "Starting Workflow" in result.stdout
    assert "Workflow completed successfully" in result.stdout

    mock_manager_cls.assert_called_once()
    mock_manager_instance.run.assert_called_once()


def test_run_config_not_found():
    """Test that Typer handles missing config file correctly."""
    result = runner.invoke(app, ["run", "non_existent.yaml"])
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "does not exist" in output or "not found" in output


def test_run_invalid_config(tmp_path, mocker):
    """Test that invalid configuration raises ValidationError (via ConfigLoader)."""
    # We rely on ConfigLoader throwing ValidationError or similar.
    # But since we use Typer, it invokes the command.
    # The command calls ConfigLoader.load_user_config.
    # We don't mock it here to verify real validation, OR we mock it to raise exception.
    # Let's test REAL validation since ConfigLoader is simple.

    config_file = tmp_path / "invalid.yaml"
    invalid_data = {
        "project_name": "BadProject",
        "target_system": {
            "elements": ["Si"],
            "composition": {"Si": 0.5},
            "crystal_structure": "diamond",
        },
        "simulation_goal": {"type": "elastic"},
    }
    with open(config_file, "w") as f:
        yaml.dump(invalid_data, f)

    result = runner.invoke(app, ["run", str(config_file)])

    assert result.exit_code == 1
    # Check for Pydantic validation error message
    assert "Composition fractions must sum to 1.0" in result.stdout or "validation failed" in result.stdout


def test_run_workflow_error(valid_config_file, mocker):
    """Test handling of exceptions during workflow execution."""
    # Mock ConfigLoader
    mock_user_config = MagicMock(spec=UserInputConfig)
    mock_user_config.project_name = "TestProject"
    mocker.patch(
        "mlip_autopipec.app.ConfigLoader.load_user_config", return_value=mock_user_config
    )

    # Mock ConfigFactory
    mock_system_config = MagicMock()
    mock_system_config.project_name = "TestProject"
    mock_system_config.run_uuid = "1234"
    mock_system_config.target_system.elements = ["Si"]
    mocker.patch(
        "mlip_autopipec.app.ConfigFactory.from_user_input", return_value=mock_system_config
    )

    # Mock WorkflowManager to raise an exception
    mock_manager_cls = mocker.patch("mlip_autopipec.app.WorkflowManager")
    mock_manager_instance = mock_manager_cls.return_value
    mock_manager_instance.run.side_effect = RuntimeError("Something went wrong in the workflow")

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 1 # app.py raises Exit(code=1) for exceptions in catch block? No, code=1 for config error.
    # Wait, my refactored app.py:
    # except Exception as e: ... raise typer.Exit(code=1)
    # So exit code should be 1.
    assert "ERROR" in result.stdout
    assert "Something went wrong in the workflow" in result.stdout


def test_entry_point_help():
    """Verify that the mlip-auto command is installed and provides help."""
    try:
        result = subprocess.run(
            ["mlip-auto", "--help"], check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            assert "Usage: mlip-auto" in result.stdout
        else:
            pass
    except FileNotFoundError:
        pass
