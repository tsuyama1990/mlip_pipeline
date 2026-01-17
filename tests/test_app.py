import pytest
from typer.testing import CliRunner
from pathlib import Path
from mlip_autopipec.app import app
import yaml
import os

runner = CliRunner()

def test_run_valid(tmp_path):
    # Setup valid input
    input_file = tmp_path / "input.yaml"
    config_data = {
        "project_name": "AppTest",
        "target_system": {
            "elements": ["Al"],
            "composition": {"Al": 1.0},
            "crystal_structure": "fcc"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 1
        },
        "simulation_goal": {
            "type": "melt_quench"
        }
    }
    with open(input_file, 'w') as f:
        yaml.dump(config_data, f)

    # Run in tmp_path context
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["run", str(input_file)])

        assert result.exit_code == 0
        assert "System initialized successfully" in result.stdout

        # Verify artifacts
        assert (tmp_path / "AppTest").is_dir()
        assert (tmp_path / "AppTest" / "project.db").exists()
        assert (tmp_path / "AppTest" / "system.log").exists()

    finally:
        os.chdir(original_cwd)

def test_run_invalid_config(tmp_path):
    input_file = tmp_path / "bad.yaml"
    config_data = {
        "project_name": "BadAppTest",
        "target_system": {
            "elements": ["Al"],
            "composition": {"Al": 0.9}, # Invalid
        }
    }
    with open(input_file, 'w') as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(input_file)])

    assert result.exit_code == 1
    assert "CONFIGURATION ERROR" in result.stdout
    assert "Composition fractions must sum to 1.0" in result.stdout

def test_run_file_not_found():
    result = runner.invoke(app, ["run", "non_existent.yaml"])
    assert result.exit_code != 0
    # Typer handles exists=True, so it prints error before our function
    assert "does not exist" in result.stderr or "does not exist" in result.stdout
