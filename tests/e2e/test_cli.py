"""End-to-end tests for CLI."""

from pathlib import Path

from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.constants import DEFAULT_CONFIG_FILENAME

runner = CliRunner()


def test_cli_init(tmp_path: Path) -> None:
    """Test 'init' command."""
    # Change to temp dir
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Created template configuration" in result.stdout
        assert Path(DEFAULT_CONFIG_FILENAME).exists()


def test_cli_init_exists(tmp_path: Path) -> None:
    """Test 'init' when file exists."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path(DEFAULT_CONFIG_FILENAME).touch()
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "already exists" in result.stdout


def test_cli_run_loop(tmp_path: Path) -> None:
    """Test 'run-loop' command."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # First init
        runner.invoke(app, ["init"])

        # Then run
        result = runner.invoke(app, ["run-loop"])
        assert result.exit_code == 0
        assert "Starting MLIP Active Learning Loop" in result.stdout

        # Check current directory
        assert Path("workflow_state.json").exists()


def test_cli_run_loop_invalid_config(tmp_path: Path) -> None:
    """Test 'run-loop' with invalid config."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create invalid config using relative path
        Path(DEFAULT_CONFIG_FILENAME).write_text("invalid: yaml: content")

        result = runner.invoke(app, ["run-loop"])
        assert result.exit_code == 1
        assert "Workflow failed" in result.stdout


def test_cli_run_loop_no_config(tmp_path: Path) -> None:
    """Test 'run-loop' without config."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run-loop"])
        assert result.exit_code == 1
        assert "not found" in result.stdout
