"""UAT for Cycle 06."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_uat_tutorial_execution(tmp_path: Path) -> None:
    """Test execution of the master tutorial script."""
    # Ensure tutorial script exists
    script_path = Path("tutorials/UAT_AND_TUTORIAL.py")
    if not script_path.exists():
        pytest.skip(f"Tutorial script not found at {script_path}. Skipping until implemented.")

    # Set environment for Mock Mode
    env = os.environ.copy()
    env["CI"] = "true"
    env["MOCK_MODE"] = "true"

    # Run the script
    # Use sys.executable for safety (S607 fixed)
    # S603: We trust the script path as it is part of the repo
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        check=False  # We manually check returncode
    )

    # Check execution success
    if result.returncode != 0:
        # Use simple string concatenation for error message instead of print
        error_msg = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        pytest.fail(f"Tutorial script failed execution.\n{error_msg}")

    # Verify artifacts (basic check)
    # Loguru writes to stderr by default
    output = result.stdout + result.stderr
    assert "Validation complete" in output
    assert "UAT Completed Successfully" in output
