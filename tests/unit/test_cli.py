from pathlib import Path

from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_init(tmp_path: Path) -> None:
    # Change to tmp_path to test file creation there
    # But typer runs in current dir usually.
    # The init command takes an output_path option.

    config_path = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--output-path", str(config_path)])
    assert result.exit_code == 0
    assert "Configuration file created" in result.stdout
    assert config_path.exists()
    assert "orchestrator:" in config_path.read_text()


def test_cli_init_exists(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    result = runner.invoke(app, ["init", "--output-path", str(config_path)])
    assert result.exit_code == 1
    # Check output (mixed) or stderr
    assert "already exists" in result.output


def test_cli_run_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_run_mock(tmp_path: Path) -> None:
    # Create valid config
    config_path = tmp_path / "config.yaml"
    runner.invoke(app, ["init", "--output-path", str(config_path)])

    # Run
    # Orchestrator run will create work_dir (default ./experiments)
    # We should override work_dir in config to be inside tmp_path,
    # OR we can just let it run. Default is ./experiments in current CWD.
    # To keep tests clean, we should update config work_dir.

    import yaml
    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    # Update work_dir to absolute path in tmp
    data["orchestrator"]["work_dir"] = str(tmp_path / "experiments")

    with config_path.open("w") as f:
        yaml.dump(data, f)

    result = runner.invoke(app, ["run", str(config_path)])

    # Capture output.
    # If successful, exit code 0.
    if result.exit_code != 0:
        print(result.stdout)

    assert result.exit_code == 0
    assert "Pipeline completed successfully" in result.stdout
    assert (tmp_path / "experiments" / "workflow_state.json").exists()
