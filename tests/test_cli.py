from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()

def test_cli_run_success(tmp_path: Path) -> None:
    config_data = {
        "workdir": str(tmp_path / "run"),
        "max_cycles": 1,
        "generator": {"type": "mock", "count": 2},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"}
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open('w') as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Pipeline completed successfully" in result.stdout
    assert (tmp_path / "run" / "potential_cycle_0.yace").exists()

def test_cli_config_not_found() -> None:
    result = runner.invoke(app, ["run", "non_existent.yaml"])
    assert result.exit_code == 1
    assert "Config file not found" in result.stderr

def test_cli_invalid_config(tmp_path: Path) -> None:
    config_data = {
        "workdir": str(tmp_path / "run"),
        # Missing max_cycles
        "generator": {"type": "mock"},
    }
    config_file = tmp_path / "bad_config.yaml"
    with config_file.open('w') as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.stderr
