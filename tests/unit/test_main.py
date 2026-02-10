from pathlib import Path

from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()

def test_init_command(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    config_file = tmp_path / "config.yaml"

    result = runner.invoke(app, ["init", "--work-dir", str(work_dir), "--config-file", str(config_file)])

    assert result.exit_code == 0
    assert work_dir.exists()
    assert config_file.exists()

def test_init_command_existing_config(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    config_file = tmp_path / "config.yaml"

    # Run once
    runner.invoke(app, ["init", "--work-dir", str(work_dir), "--config-file", str(config_file)])

    # Run again
    result = runner.invoke(app, ["init", "--work-dir", str(work_dir), "--config-file", str(config_file)])

    assert result.exit_code == 0
    assert "already exists" in result.stdout

def test_run_loop_command(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    config_file = tmp_path / "config.yaml"

    # Init
    runner.invoke(app, ["init", "--work-dir", str(work_dir), "--config-file", str(config_file)])

    # Run loop
    result = runner.invoke(app, ["run-loop", "--config-file", str(config_file)])

    assert result.exit_code == 0
    # Verify iteration output in stdout?
    # Wait, invoke(app) captures output?
    # Typer/Click captures stdout/stderr.
    # But logger logs to console using Rich handler.
    # Rich handler prints to console.
    # CliRunner might capture it if configured correctly.
    # But log file should exist.
    assert (Path("mlip_pipeline.log")).exists() or (work_dir / "mlip_pipeline.log").exists()
    # Actually setup_logging uses default log file "mlip_pipeline.log" in current dir unless changed.
    # Let's check state file
    assert (work_dir / "workflow_state.json").exists()

def test_run_loop_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run-loop", "--config-file", "non_existent.yaml"])
    assert result.exit_code == 1
    assert "Error" in result.stdout or "Error" in result.stderr
