"""User Acceptance Tests for Cycle 01."""

import os
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
    # Ensure config matches schema (version required)
    config_content = f"""
version: "0.1.0"
project:
  name: "TestProject"
  root_dir: "{tmp_path.resolve()}"
oracle:
  dft:
    code: "quantum_espresso"
    pseudopotentials:
      Fe: Fe.pbe.UPF
  mock: true
trainer:
  mock: true
"""
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(config_content)

    # Need to set PYTHONPATH to include src
    env = {"PYTHONPATH": "src"}
    # Also skip file checks to avoid missing pseudopotential error
    env["PYACEMAKER_SKIP_FILE_CHECKS"] = "true"

    env.update(os.environ)

    result = subprocess.run(
        ["pyacemaker", "run", str(config_file)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
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

    env = {"PYTHONPATH": "src"}
    env.update(os.environ)

    result = subprocess.run(
        ["pyacemaker", "run", str(config_file)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    # Error message usually goes to stderr
    assert (
        "Field required" in result.stderr
        or "Validation" in result.stderr
        or "Error" in result.stderr
    )
