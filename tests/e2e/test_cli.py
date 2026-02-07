from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Run the MLIP pipeline" in result.stdout


def test_cli_missing_config() -> None:
    result = runner.invoke(app, ["run", "nonexistent.yaml"])
    assert result.exit_code == 1
    # Check stdout/stderr depending on implementation
    assert "Config file not found" in result.stdout or "Config file not found" in result.stderr


def test_cli_invalid_config(tmp_path) -> None:
    p = tmp_path / "invalid.yaml"
    p.write_text("oracle:\n  type: unknown")
    result = runner.invoke(app, ["run", str(p)])
    assert result.exit_code == 1
    assert "Error loading config" in result.stdout or "Error loading config" in result.stderr


def test_cli_valid_run(tmp_path) -> None:
    # Use subdirectory for workdir to avoid cluttering tmp_path root
    workdir = tmp_path / "work"
    config_content = f"""
    workdir: "{workdir}"
    oracle:
      type: "mock"
      noise_level: 0.1
    trainer:
      type: "mock"
    dynamics:
      type: "mock"
    """

    p = tmp_path / "config.yaml"
    p.write_text(config_content)

    result = runner.invoke(app, ["run", str(p)])
    # Check exit code
    assert result.exit_code == 0
    # Check output from logger
    # Note: runner.invoke captures stdout/stderr.
    # Our logger writes to stdout.
    assert "Configuration loaded successfully" in result.stdout
    assert "Initialised MockOrchestrator" in result.stdout
    assert "Pipeline finished" in result.stdout
