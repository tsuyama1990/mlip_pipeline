import logging
from pathlib import Path

from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()

def test_cli_help() -> None:
    """Test that the main help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MLIP Pipeline Runner" in result.output

def test_cli_run_help() -> None:
    """Test that the run subcommand help works."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run the MLIP active learning workflow" in result.output

def test_cli_run_missing_config() -> None:
    """Test run command with a non-existent config file."""
    result = runner.invoke(app, ["run", "missing.yaml"])
    assert result.exit_code == 1
    assert "Error: Config file missing.yaml not found." in result.output

def test_cli_run_invalid_config(tmp_path: Path) -> None:
    """Test run command with an invalid config file."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid_field: value")

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    assert "Error loading configuration" in result.output

def test_cli_run_success(tmp_path: Path) -> None:
    """Test run command with a valid config."""
    # Reset logger handlers to force file handler creation
    logger = logging.getLogger("mlip_autopipec")
    logger.handlers = []

    work_dir = tmp_path / "test_run"
    config_content = f"""
    orchestrator:
      work_dir: {work_dir}
      max_cycles: 1
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0

    # Verify the log file was created
    log_file = work_dir / "mlip.log"
    assert log_file.exists()
