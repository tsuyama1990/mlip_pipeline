import pytest
from typer.testing import CliRunner
from mlip_autopipec.app import app
import yaml
import ase.db
import os
from pathlib import Path

runner = CliRunner()

def test_uat_01_01_valid_initialization(tmp_path):
    """
    UAT-01-01: Valid Project Initialization
    Verify that a user can initialize a new project by providing a valid input.yaml.
    """
    # GIVEN
    input_file = tmp_path / "input.yaml"
    config_data = {
        "project_name": "AlCu_Alloy",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.5},
            "crystal_structure": "fcc"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        },
        "simulation_goal": {
            "type": "melt_quench"
        }
    }
    with open(input_file, 'w') as f:
        yaml.dump(config_data, f)

    # WHEN
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["run", str(input_file)])

        # THEN
        if result.exit_code != 0:
            print(result.stdout)
            print(result.stderr)

        assert result.exit_code == 0
        assert "System initialized successfully" in result.stdout

        project_dir = tmp_path / "AlCu_Alloy"
        assert project_dir.is_dir()
        assert (project_dir / "project.db").exists()
        assert (project_dir / "system.log").exists()

        with open(project_dir / "system.log") as f:
            assert "Logging initialized" in f.read()

    finally:
        os.chdir(cwd)

def test_uat_01_02_invalid_configuration(tmp_path):
    """
    UAT-01-02: Invalid Configuration Handling
    Verify that the system provides clear error messages for invalid configuration.
    """
    # GIVEN
    input_file = tmp_path / "bad_input.yaml"
    config_data = {
        "project_name": "AlCu_Alloy",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5}, # Missing Cu, sum != 1.0
            "crystal_structure": "fcc"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        },
        "simulation_goal": {
            "type": "melt_quench"
        }
    }
    with open(input_file, 'w') as f:
        yaml.dump(config_data, f)

    # WHEN
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["run", str(input_file)])

        # THEN
        assert result.exit_code == 1
        assert "CONFIGURATION ERROR" in result.stdout
        assert "Composition fractions must sum to 1.0" in result.stdout

        # Ensure no project created
        project_dir = tmp_path / "AlCu_Alloy"
        assert not project_dir.exists()

    finally:
        os.chdir(cwd)

def test_uat_01_03_db_provenance(tmp_path):
    """
    UAT-01-03: Database & Provenance Check
    Verify that the initialized database contains the configuration settings in its metadata.
    """
    # GIVEN
    input_file = tmp_path / "input.yaml"
    config_data = {
        "project_name": "ProvenanceTest",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.5},
            "crystal_structure": "fcc"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        },
        "simulation_goal": {
            "type": "melt_quench"
        }
    }
    with open(input_file, 'w') as f:
        yaml.dump(config_data, f)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        runner.invoke(app, ["run", str(input_file)])

        # WHEN
        db_path = tmp_path / "ProvenanceTest" / "project.db"
        with ase.db.connect(db_path) as db:
            meta = db.metadata

        # THEN
        assert meta is not None
        assert meta['minimal']['target_system']['elements'] == ['Al', 'Cu']
        assert meta['working_dir'] == str(tmp_path / "ProvenanceTest")

    finally:
        os.chdir(cwd)
