import pytest
from typer.testing import CliRunner
from mlip_autopipec.app import app
import yaml
import ase.db

runner = CliRunner()

def test_uat_01_01_valid_project_initialization(tmp_path):
    """
    UAT-01-01: Valid Project Initialization
    Verify that a user can initialize a new project by providing a valid input.yaml.
    """
    input_file = tmp_path / "input.yaml"
    config_data = {
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
    input_file.write_text(yaml.dump(config_data))

    # Run in tmp_path
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run", str(input_file)])

        assert result.exit_code == 0, f"STDOUT: {result.stdout}"
        assert "System initialized successfully" in result.stdout

        project_dir = tmp_path / "AlCu_Alloy"
        assert project_dir.exists()
        assert (project_dir / "project.db").exists()
        assert (project_dir / "system.log").exists()

def test_uat_01_02_invalid_configuration(tmp_path):
    """
    UAT-01-02: Invalid Configuration Handling
    Verify that the system provides clear error messages for invalid configuration.
    """
    input_file = tmp_path / "bad_input.yaml"
    config_data = {
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
    input_file.write_text(yaml.dump(config_data))

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run", str(input_file)])

        assert result.exit_code != 0
        assert "Composition fractions must sum to 1.0" in result.stdout

def test_uat_01_03_database_metadata(tmp_path):
    """
    UAT-01-03: Database & Provenance Check
    Verify that the initialized database contains the configuration settings in its metadata.
    """
    input_file = tmp_path / "input.yaml"
    config_data = {
        "project_name": "MetaTest",
        "target_system": {
            "elements": ["Al"],
            "composition": {"Al": 1.0}
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 1
        }
    }
    input_file.write_text(yaml.dump(config_data))

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run", str(input_file)])
        assert result.exit_code == 0

        db_path = tmp_path / "MetaTest" / "project.db"
        assert db_path.exists()

        # Connect using ase.db and check metadata
        with ase.db.connect(str(db_path)) as db:
            meta = db.metadata

        assert "system_config" in meta
        assert meta["system_config"]["minimal"]["project_name"] == "MetaTest"
        assert meta["system_config"]["minimal"]["target_system"]["elements"] == ["Al"]
