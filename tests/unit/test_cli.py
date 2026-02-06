from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MLIP Automated Pipeline CLI" in result.stdout


def test_cli_run_missing_config() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "Missing option" in result.stdout or "Missing option" in result.stderr


def test_cli_run_nonexistent_config() -> None:
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code == 1
    # Check log output? Typer runner captures stdout/stderr.
    # We used setup_logging which logs to stdout.
    # Typer runner should capture it.
    assert "Config file not found" in result.stderr or "Config file not found" in result.stdout


def test_cli_run_invalid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content")

    result = runner.invoke(app, ["run", "--config", str(config_file)])
    assert result.exit_code == 1
    # We use logging.exception which prints traceback
    assert "Error parsing config" in result.stdout or "Error parsing config" in result.stderr


def test_cli_run_success(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data = {
        "work_dir": str(tmp_path / "work"),
        "max_cycles": 1,
        "random_seed": 42
    }
    with config_file.open("w") as f:
        yaml.dump(data, f)

    result = runner.invoke(app, ["run", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "Pipeline completed" in result.stdout
