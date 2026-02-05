from pathlib import Path

import yaml
from typer.testing import CliRunner

from src.main import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Run the MLIP active learning pipeline" in result.stdout


def test_cli_run_missing_config() -> None:
    # Use --config option
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code == 1, f"Exit code {result.exit_code}. Output: {result.stdout}"
    assert "Config file not found" in result.stdout or (
        result.stderr and "Config file not found" in result.stderr
    )


def test_cli_run_valid(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_data = {
        "work_dir": str(tmp_path / "workspace"),
        "max_cycles": 1,
        "random_seed": 42,
    }
    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    # Use --config option
    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code == 0, f"Exit code {result.exit_code}. Output: {result.stdout}"
