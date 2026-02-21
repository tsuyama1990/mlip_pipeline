"""User Acceptance Tests for Cycle 01."""

import os
import subprocess
import sys
from pathlib import Path


def test_uat_cli_help() -> None:
    """Scenario 01: CLI Health Check."""
    env = {"PYTHONPATH": "src"}
    env.update(os.environ)

    # Force use of venv python to ensure deps are found
    python_exe = (
        "/app/.venv/bin/python3" if Path("/app/.venv/bin/python3").exists() else sys.executable
    )

    # Running via -m fails to capture output with subprocess sometimes due to typer/rich interaction?
    # Or maybe it's just how the test environment captures stdout.
    # Using -c to invoke explicitly seems robust.

    cmd = [
        python_exe,
        "-c",
        "import sys; sys.argv=['pyacemaker', '--help']; from pyacemaker.main import app; app()",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "Usage" in combined


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

    # Use python -m pyacemaker.main instead of command alias to ensure imports work in test env
    # without full install.
    # Actually, pyacemaker command might not be found if not installed in editable mode or path.
    # Safest is to run via python -m

    # But wait, pyacemaker.main calls app(). We need to import it properly.
    # The entry point is defined as pyacemaker = pyacemaker.main:app

    # Let's try running as module
    # Force use of venv python
    python_exe = (
        "/app/.venv/bin/python3" if Path("/app/.venv/bin/python3").exists() else sys.executable
    )

    cmd = [
        python_exe,
        "-c",
        f"import sys; sys.argv=['pyacemaker', 'run', r'{config_file}']; from pyacemaker.main import app; app()",
    ]

    result = subprocess.run(
        cmd,
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

    # Force use of venv python
    python_exe = (
        "/app/.venv/bin/python3" if Path("/app/.venv/bin/python3").exists() else sys.executable
    )

    cmd = [
        python_exe,
        "-c",
        f"import sys; sys.argv=['pyacemaker', 'run', r'{config_file}']; from pyacemaker.main import app; app()",
    ]

    result = subprocess.run(
        cmd,
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
