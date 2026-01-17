import ase.db
import pytest
import yaml


# Helpers
def run_command(cmd, cwd=None):
    import subprocess
    return subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True)

@pytest.fixture
def uat_work_dir(tmp_path):
    work_dir = tmp_path / "uat_cycle_01"
    work_dir.mkdir()
    return work_dir

@pytest.mark.parametrize("project_suffix", ["Alpha", "Beta", "Gamma"])
def test_uat_01_valid_initialization(uat_work_dir, project_suffix):
    """UAT-01-01: Valid Project Initialization with multiple inputs."""
    project_name = f"TestProject_{project_suffix}"
    input_file = uat_work_dir / f"input_{project_suffix}.yaml"
    input_data = {
        "project_name": project_name,
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
    project_dir = uat_work_dir / project_name
    assert project_dir.exists()
    assert (project_dir / "project.db").exists()
    assert (project_dir / "system.log").exists()

    # Check database metadata (UAT-01-03)
    db = ase.db.connect(str(project_dir / "project.db"))
    db.count() # Force initialization
    metadata = db.metadata
    assert metadata is not None
    assert metadata["minimal"]["project_name"] == project_name

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

def test_uat_04_idempotency(uat_work_dir):
    """UAT-01-04: Idempotency Check"""
    project_name = "IdemProject"
    input_file = uat_work_dir / "idem_input.yaml"
    input_data = {
        "project_name": project_name,
        "target_system": {
            "elements": ["Ni"],
            "composition": {"Ni": 1.0}
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 1
        }
    }
    with input_file.open("w") as f:
        yaml.dump(input_data, f)

    # Run twice
    res1 = run_command(["mlip-auto", "run", str(input_file)], cwd=uat_work_dir)
    assert res1.returncode == 0

    res2 = run_command(["mlip-auto", "run", str(input_file)], cwd=uat_work_dir)
    assert res2.returncode == 0 # Should succeed essentially, maybe warn log
    # We check if DB is still valid
    project_dir = uat_work_dir / project_name
    db = ase.db.connect(str(project_dir / "project.db"))
    db.count() # Force initialization
    assert db.metadata["minimal"]["project_name"] == project_name
