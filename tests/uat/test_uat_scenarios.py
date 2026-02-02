import os
import subprocess
import sys
from pathlib import Path


def test_uat_01_01_first_breath(temp_project_dir: Path, valid_config_yaml: Path, dummy_dataset: Path) -> None:
    """
    Scenario 01-01: The First Breath
    Verify that the system can initialize, read configuration, and complete a single "Skeleton Cycle" without crashing.
    """
    # We need to run the main module.
    # We set CWD to the project dir.

    # We also need to enforce Mock mode via env var
    env = {"PYACEMAKER_MOCK_MODE": "1", "PATH": os.environ["PATH"]}

    # Run command
    # python -m mlip_autopipec.main config.yaml

    cmd = [sys.executable, "-m", "mlip_autopipec.main", str(valid_config_yaml.name)]

    result = subprocess.run(
        cmd,
        cwd=temp_project_dir,
        capture_output=True,
        text=True,
        env=env,
        check=False
    )

    # Debug output if fail
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "Workflow completed successfully" in result.stdout or "Workflow completed" in result.stdout
    assert "ERROR" not in result.stderr

    # Check artifacts
    assert (temp_project_dir / "workflow_state.json").exists()
    assert (temp_project_dir / "output_potential.yace").exists()


def test_uat_01_02_guard_rails(temp_project_dir: Path) -> None:
    """
    Scenario 01-02: The Guard Rails
    Verify that the system provides helpful feedback when the configuration is invalid.
    """
    bad_config = temp_project_dir / "bad_config.yaml"
    bad_config.write_text("""
project:
  name: "Bad"
training:
  dataset_path: "missing.pckl"
""")

    cmd = [sys.executable, "-m", "mlip_autopipec.main", "bad_config.yaml"]

    result = subprocess.run(
        cmd,
        cwd=temp_project_dir,
        capture_output=True,
        text=True,
        check=False
    )

    assert result.returncode != 0
    # Can be generic "does not exist" or specific path error or pydantic error
    assert any(msg in result.stderr for msg in ["does not exist", "not found", "Path does not point to a file"])
