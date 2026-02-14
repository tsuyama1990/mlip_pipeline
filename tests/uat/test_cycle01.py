"""User Acceptance Tests for Cycle 01."""

import subprocess
from pathlib import Path


def test_uat_cli_help() -> None:
    """Scenario 01: CLI Health Check."""
    result = subprocess.run(
        ["pyacemaker", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Usage" in result.stdout


def test_uat_valid_config(tmp_path: Path) -> None:
    """Scenario 02: Valid Configuration Loading."""
    config_content = """
project:
  name: "TestProject"
  root_dir: "./"
oracle:
  dft:
    code: "quantum_espresso"
"""
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(config_content)

    result = subprocess.run(
        ["pyacemaker", "run", str(config_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (
        "Configuration loaded successfully" in result.stdout
        or "Configuration loaded successfully" in result.stderr
    )


def test_uat_invalid_config(tmp_path: Path) -> None:
    """Scenario 03: Invalid Configuration Handling."""
    config_content = """
project:
  # Missing name
  root_dir: "./"
"""
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text(config_content)

    result = subprocess.run(
        ["pyacemaker", "run", str(config_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    # Error message usually goes to stderr
    assert (
        "Field required" in result.stderr
        or "Validation" in result.stderr
        or "Error" in result.stderr
    )
