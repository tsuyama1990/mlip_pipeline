from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_run_mock(tmp_path: Path) -> None:
    """Test the CLI 'run' command with a valid mock configuration."""
    # Create config file
    config_path = tmp_path / "config.yaml"
    config_data = {
        "workdir": str(tmp_path / "runs"),
        "max_cycles": 1,
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0
    assert "Starting pipeline" in result.stdout
    assert "Cycle 1" in result.stdout


def test_cli_invalid_config_path() -> None:
    """Test CLI with a non-existent configuration file."""
    result = runner.invoke(app, ["run", "non_existent.yaml"])
    assert result.exit_code == 2
    # Typer catches this due to exists=True


def test_cli_validation_error(tmp_path: Path) -> None:
    """Test CLI with a configuration that fails Pydantic validation."""
    config_path = tmp_path / "invalid_config.yaml"
    # Missing required fields
    config_data = {"max_cycles": 1}
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 1
    # Check if this message is in stdout (since I use logger.error which goes to stderr? No, logger uses stdout in setup_logging)
    # But wait, setup_logging uses sys.stdout.
    # However, runner captures stdout.
    assert "Configuration validation error" in result.stdout
