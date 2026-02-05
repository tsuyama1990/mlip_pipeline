from typer.testing import CliRunner
from mlip_autopipec.main import app
import pytest
from mlip_autopipec.utils.logging import setup_logging
import logging

runner = CliRunner()

def test_setup_logging(capsys) -> None:
    setup_logging()
    logger = logging.getLogger("test_logger")
    logger.info("Test logging message")

    captured = capsys.readouterr()
    assert "Test logging message" in captured.out

def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    if result.exit_code != 0 or "MLIP Pipeline CLI" not in result.stdout:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.exit_code == 0
    assert "MLIP Pipeline CLI" in result.stdout

def test_cli_run_missing_config() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "Missing option" in result.stderr or "Missing option" in result.stdout

def test_cli_run_nonexistent_config() -> None:
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    if result.exit_code != 1:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.exit_code == 1
    assert "not found" in result.stderr or "not found" in result.stdout

def test_cli_run_invalid_yaml(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("invalid_yaml: [")
    result = runner.invoke(app, ["run", "--config", str(config)])
    if result.exit_code != 1:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.exit_code == 1
    assert "Error reading YAML file" in result.stderr or "Error reading YAML file" in result.stdout

def test_cli_run_invalid_schema(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("max_cycles: -1\nwork_dir: ./workspace")
    result = runner.invoke(app, ["run", "--config", str(config)])
    if result.exit_code != 1:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.exit_code == 1
    assert "Error parsing config" in result.stderr or "Error parsing config" in result.stdout

def test_cli_run_valid_config(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("""
work_dir: ./workspace
max_cycles: 1
random_seed: 42
    """)
    result = runner.invoke(app, ["run", "--config", str(config)])
    if result.exit_code != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.exit_code == 0
