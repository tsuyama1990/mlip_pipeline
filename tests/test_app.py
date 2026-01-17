import subprocess
import pytest
import yaml
from typer.testing import CliRunner
from unittest.mock import MagicMock

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
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return config_file

def test_run_success(valid_config_file, mocker, tmp_path):
    """Test the happy path: CLI calls ConfigFactory and DatabaseManager."""
    mock_factory = mocker.patch("mlip_autopipec.app.ConfigFactory")
    mock_db_manager = mocker.patch("mlip_autopipec.app.DatabaseManager")
    mock_logging = mocker.patch("mlip_autopipec.app.setup_logging")

    # Mock return value of factory
    mock_config = MagicMock()
    mock_config.working_dir = tmp_path / "TestProject"
    mock_config.db_path = mock_config.working_dir / "project.db"
    mock_config.log_path = mock_config.working_dir / "system.log"
    mock_factory.from_yaml.return_value = mock_config

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 0
    assert "System initialized successfully" in result.stdout

    mock_factory.from_yaml.assert_called_once()
    mock_logging.assert_called_once_with(mock_config.log_path)
    mock_db_manager.assert_called_once_with(mock_config.db_path)
    mock_db_manager.return_value.initialize.assert_called_once_with(mock_config)

def test_run_failure(valid_config_file, mocker):
    """Test that CLI handles errors gracefully."""
    mock_factory = mocker.patch("mlip_autopipec.app.ConfigFactory")
    mock_factory.from_yaml.side_effect = ValueError("Something exploded")

    result = runner.invoke(app, ["run", str(valid_config_file)])

    assert result.exit_code == 1
    assert "FAILURE" in result.stdout
    assert "Something exploded" in result.stdout

def test_entry_point_help():
    """Verify that the mlip-auto command provides help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage: " in result.stdout
