import subprocess

import ase.db
import pytest
import yaml


# Helpers
def run_command(cmd, cwd=None):
    return subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True)

@pytest.fixture
def uat_work_dir(tmp_path):
    work_dir = tmp_path / "uat_cycle_01"
    work_dir.mkdir()
    return work_dir
    # Cleanup done by tmp_path

def test_uat_01_valid_initialization(uat_work_dir):
    """UAT-01-01: Valid Project Initialization"""
    input_file = uat_work_dir / "input.yaml"
    input_data = {
        "project_name": "AlCu_Alloy",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.5}
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        }
    }
    with input_file.open("w") as f:
        yaml.dump(input_data, f)

    # Run command
    res = run_command(["mlip-auto", "run", str(input_file)], cwd=uat_work_dir)

    assert res.returncode == 0
    assert "System initialized" in res.stdout or "System initialized" in res.stderr

    # Check directory structure
    project_dir = uat_work_dir / "AlCu_Alloy"
    assert project_dir.exists()
    assert (project_dir / "project.db").exists()
    assert (project_dir / "system.log").exists()

    # Check database metadata (UAT-01-03)
    db = ase.db.connect(str(project_dir / "project.db"))
    db.count() # Force initialization
    metadata = db.metadata
    assert metadata is not None
    # We might need to look deeper depending on how we saved it
    # SystemConfig.model_dump() usually creates nested dicts
    assert metadata["minimal"]["project_name"] == "AlCu_Alloy"
    assert metadata["minimal"]["target_system"]["elements"] == ["Al", "Cu"]

def test_uat_02_invalid_configuration(uat_work_dir):
    """UAT-01-02: Invalid Configuration Handling"""
    input_file = uat_work_dir / "bad_input.yaml"
    input_data = {
        "project_name": "BadProject",
        "target_system": {
            "elements": ["Fe"],
            "composition": {"Fe": 0.9} # Invalid sum
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        }
    }
    with input_file.open("w") as f:
        yaml.dump(input_data, f)

    res = run_command(["mlip-auto", "run", str(input_file)], cwd=uat_work_dir)

    assert res.returncode != 0
    assert "Composition fractions must sum to 1.0" in res.stdout
