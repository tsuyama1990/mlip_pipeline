import pytest
import subprocess
import yaml
from pathlib import Path

def test_uat_01_02_mock_loop_execution(tmp_path: Path) -> None:
    """
    UAT-01-02: Verify that the orchestrator runs through N cycles using Mock components.
    """
    # GIVEN a configuration file with execution_mode: mock
    config_data = {
        "execution_mode": "mock",
        "max_cycles": 3,
        "project_name": "uat_test_project",
        "exploration": {"max_structures": 2},
        "dft": {"calculator": "espresso"},
        "training": {"potential_type": "ace"}
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # WHEN the user runs mlip-pipeline run config.yaml
    cmd = ["uv", "run", "mlip-pipeline", "run", str(config_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # THEN the system should exit with success
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # AND output should contain key milestones
    stdout = result.stdout
    assert "Starting Cycle 1" in stdout
    assert "MockExplorer generated" in stdout
    assert "MockOracle calculated" in stdout
    assert "MockTrainer updated potential" in stdout
    assert "MockValidator validating" in stdout
    assert "Starting Cycle 2" in stdout
    assert "Starting Cycle 3" in stdout
    assert "Workflow completed successfully" in stdout

def test_uat_01_03_config_validation(tmp_path: Path) -> None:
    """
    UAT-01-03: Verify that the system rejects a malformed config.yaml.
    """
    # GIVEN a malformed config
    config_data = {
        "execution_mode": "mock",
        "max_cycles": "invalid_number" # Invalid type
    }
    config_file = tmp_path / "bad_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # WHEN the user runs mlip-pipeline
    cmd = ["uv", "run", "mlip-pipeline", "run", str(config_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # THEN the system should exit with error
    assert result.returncode == 1
    # AND print validation error
    assert "Configuration Validation Error" in result.stderr or "Configuration Validation Error" in result.stdout
