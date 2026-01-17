import pytest
from typer.testing import CliRunner
from mlip_autopipec.app import app
from ase.db import connect
import yaml

runner = CliRunner()

RESOURCES_DEFAULT = {
    "dft_code": "quantum_espresso",
    "parallel_cores": 4,
    "gpu_enabled": False
}

def test_uat_01_01_valid_initialization(tmp_path, mocker):
    """UAT-01-01: Valid Project Initialization"""
    input_file = tmp_path / "input.yaml"
    data = {
        "project_name": "AlCu_Alloy",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.5},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
        "resources": RESOURCES_DEFAULT
    }
    with input_file.open("w") as f:
        yaml.dump(data, f)

    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)

    result = runner.invoke(app, ["init", str(input_file)])

    assert result.exit_code == 0
    assert "SUCCESS" in result.stdout

    project_dir = tmp_path / "AlCu_Alloy"
    assert project_dir.exists()
    assert (project_dir / "AlCu_Alloy.db").exists()
    assert (project_dir / "system.log").exists()

def test_uat_01_02_invalid_configuration(tmp_path):
    """UAT-01-02: Invalid Configuration Handling"""
    input_file = tmp_path / "bad_input.yaml"
    data = {
        "project_name": "BadProject",
        "target_system": {
            "elements": ["Fe"],
            "composition": {"Fe": 0.9}, # Sum != 1.0
            "crystal_structure": "bcc",
        },
        "simulation_goal": {"type": "melt_quench"},
        "resources": RESOURCES_DEFAULT
    }
    with input_file.open("w") as f:
        yaml.dump(data, f)

    result = runner.invoke(app, ["init", str(input_file)])

    assert result.exit_code == 1
    assert "CONFIGURATION ERROR" in result.stdout
    assert "Composition fractions must sum to 1.0" in result.stdout

def test_uat_01_03_database_provenance(tmp_path, mocker):
    """UAT-01-03: Database & Provenance Check"""
    input_file = tmp_path / "input_prov.yaml"
    data = {
        "project_name": "ProvTest",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.5},
            "crystal_structure": "fcc",
        },
        "simulation_goal": {"type": "melt_quench"},
        "resources": RESOURCES_DEFAULT
    }
    with input_file.open("w") as f:
        yaml.dump(data, f)

    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)
    runner.invoke(app, ["init", str(input_file)])

    db_path = tmp_path / "ProvTest" / "ProvTest.db"

    with connect(db_path) as conn:
        meta = conn.metadata
        # Check metadata content
        assert "user_input" in meta
        assert meta["user_input"]["target_system"]["elements"] == ["Al", "Cu"]
        # Check absolute path
        # Note: metadata is serialized to JSON, paths might be strings.
        assert str(tmp_path) in meta["working_dir"]
