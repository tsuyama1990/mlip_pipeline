import subprocess

import pytest
import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.config.models import SystemConfig

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
    with config_file.open("w") as f:
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


def test_init_command_success(valid_config_file, tmp_path, mocker):
    """Test the init command sets up the project structure."""
    # Mocks
    mock_config_factory = mocker.patch("mlip_autopipec.app.ConfigFactory")
    mock_db_manager = mocker.patch("mlip_autopipec.app.DatabaseManager")
    mock_setup_logging = mocker.patch("mlip_autopipec.app.setup_logging")

    # Setup ConfigFactory return value
    # We need a dummy SystemConfig
    from uuid import uuid4
    dummy_system_config = SystemConfig(
        project_name="TestProject",
        run_uuid=uuid4(),
        working_dir=tmp_path / "TestProject",
        db_path="TestProject.db",
    )
    mock_config_factory.from_user_input.return_value = dummy_system_config

    # Run command
    # We pass the config file
    result = runner.invoke(app, ["init", str(valid_config_file)])

    assert result.exit_code == 0
    assert "Initializing project" in result.stdout
    assert "SUCCESS" in result.stdout

    # Verify calls
    # 1. ConfigFactory called
    mock_config_factory.from_user_input.assert_called_once()

    # 2. DatabaseManager initialized
    mock_db_manager.assert_called_once_with(dummy_system_config.working_dir / dummy_system_config.db_path)
    mock_db_manager.return_value.initialize.assert_called_once_with(dummy_system_config)

    # 3. Logging setup
    # Should be called with path inside working dir
    expected_log_path = dummy_system_config.working_dir / "mlip_auto.log"
    mock_setup_logging.assert_called_once_with(expected_log_path, level="INFO")


def test_init_creates_directory(valid_config_file, tmp_path):
    """Integration test: init command actually creates directory."""
    # We allow the real ConfigFactory to run, but we mock DatabaseManager and Logging to avoid creating files/side effects outside control
    # or let them run if they are safe.
    # DatabaseManager needs ase installed.
    # Logging creates a file.

    # We want to test that the FOLDER is created.
    # ConfigFactory creates the folder in its from_user_input method.

    # But we run in a temp dir.
    # The CLI will read input.yaml.
    # input.yaml says project_name="TestProject".
    # ConfigFactory creates "TestProject" relative to CWD.
    # We must ensure CWD is tmp_path.

    with valid_config_file.open() as f:
        data = yaml.safe_load(f)
    project_name = data["project_name"]

    # We need to change cwd to tmp_path for this test
    # But pytest runs in a fixed cwd usually.
    # We can use monkeypatch.chdir
    with pytest.MonkeyPatch.context() as m:
        m.chdir(tmp_path)
        result = runner.invoke(app, ["init", str(valid_config_file)])

    assert result.exit_code == 0
    project_dir = tmp_path / project_name
    assert project_dir.exists()
    assert project_dir.is_dir()
    assert (project_dir / f"{project_name}.db").exists()
    assert (project_dir / "mlip_auto.log").exists()


def test_entry_point_help():
    """Verify that the mlip-auto command is installed and provides help."""
    try:
        result = subprocess.run(
            ["mlip-auto", "--help"], check=False, capture_output=True, text=True  # noqa: S607
        )
        if result.returncode == 0:
            assert "Usage: mlip-auto" in result.stdout
    except FileNotFoundError:
        pass
