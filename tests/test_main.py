"""Tests for the CLI entry point."""

from pathlib import Path

from typer.testing import CliRunner

from pyacemaker.main import app

runner = CliRunner()


def test_app_help() -> None:
    """Test that the CLI help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout
    assert "Options" in result.stdout


def test_run_command_valid(tmp_path: Path) -> None:
    """Test running the pipeline with a valid configuration."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "project:\n  name: Test\n  root_dir: .\noracle:\n  dft:\n    code: qe\n    pseudopotentials:\n      Fe: Fe.pbe.UPF",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Configuration loaded successfully" in result.stdout


def test_run_command_invalid_config(tmp_path: Path) -> None:
    """Test running with an invalid configuration."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml", encoding="utf-8")

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    # Should catch PYACEMAKERError and print Error: ...
    assert "Error" in result.stderr


def test_run_command_missing_file() -> None:
    """Test running with a non-existent configuration file."""
    result = runner.invoke(app, ["run", "non_existent.yaml"])
    # Typer catches file not found if exists=True is set on argument
    assert result.exit_code != 0
    assert "does not exist" in result.stderr or "does not exist" in result.stdout
