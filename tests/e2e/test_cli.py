from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()

def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # The help message should contain the app help description
    assert "PYACEMAKER" in result.stdout

def test_cli_run_mock_config(tmp_path: Path) -> None:
    # Create valid config
    config_data = {
        "workdir": str(tmp_path / "work"),
        "max_cycles": 1,
        "oracle": {"type": "mock", "noise_level": 0.05},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"}
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])

    assert result.exit_code == 0
    assert "Configuration loaded successfully" in result.stdout or "Configuration loaded successfully" in result.stderr

def test_cli_run_invalid_config(tmp_path: Path) -> None:
    config_data = {
        "workdir": str(tmp_path / "work"),
        "oracle": {"type": "unknown"}
    }
    config_file = tmp_path / "invalid.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])

    assert result.exit_code == 1
    assert "Validation Error" in result.stderr
