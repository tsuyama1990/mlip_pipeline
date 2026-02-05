import pytest
from typer.testing import CliRunner
from mlip_autopipec.main import app
from pathlib import Path
import yaml

runner = CliRunner()

def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MLIP Pipeline CLI" in result.stdout

def test_cli_run_mock(tmp_path: Path) -> None:
    # Create config
    config_data = {
        "execution_mode": "mock",
        "max_cycles": 1,
        "project_name": "test_cli"
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Initializing MOCK components" in result.stdout
    assert "Workflow completed successfully" in result.stdout

def test_cli_run_invalid_config(tmp_path: Path) -> None:
    # Invalid config (missing fields or wrong type if strict, but GlobalConfig has defaults)
    # Let's pass invalid type
    config_data = {
        "max_cycles": "invalid"
    }
    config_file = tmp_path / "bad_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    assert "Configuration Validation Error" in result.stdout

def test_cli_not_implemented_real(tmp_path: Path) -> None:
    config_data = {
        "execution_mode": "real"
    }
    config_file = tmp_path / "real_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    assert "not yet supported" in result.stdout
