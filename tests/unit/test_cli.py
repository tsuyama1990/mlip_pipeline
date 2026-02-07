from pathlib import Path

from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_init(tmp_path: Path) -> None:
    # Use tmp_path directly without isolated_filesystem for simplicity if path is absolute,
    # but CliRunner works well with isolation.
    # We can pass path explicitly.
    config_path = tmp_path / "test_config.yaml"
    result = runner.invoke(app, ["init", "--path", str(config_path)])
    assert result.exit_code == 0
    assert config_path.exists()
    assert "Generated default config" in result.stdout


def test_cli_run(tmp_path: Path) -> None:
    # 1. Init config
    config_path = tmp_path / "config.yaml"
    runner.invoke(app, ["init", "--path", str(config_path)])

    # 2. Run
    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "Initialized MockOracle" in result.stdout
    assert "Workdir" in result.stdout


def test_cli_run_missing_config() -> None:
    result = runner.invoke(app, ["run", "--config", "non_existent.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_cli_run_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text("invalid_yaml: [")

    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.stdout
