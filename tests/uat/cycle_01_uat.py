
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app

# UAT Scenarios for Cycle 1
# Reference: dev_documents/system_prompts/CYCLE01/UAT.md  # noqa: ERA001

runner = CliRunner()

@pytest.fixture
def uat_work_dir(tmp_path: Path) -> Path:
    # Use a subdirectory in tmp_path to avoid clutter
    d = tmp_path / "uat_cycle01"
    d.mkdir()
    return d

def test_uat_01_01_valid_initialization(uat_work_dir: Path) -> None:
    """
    UAT-01-01: Valid Project Initialization
    Verify that a user can initialize a new project by providing a valid input.yaml.
    """
    # GIVEN a clean working directory (uat_work_dir is clean)
    # AND a valid input.yaml
    input_yaml_path = uat_work_dir / "input.yaml"
    input_data = {
        "project_name": "AlCu_Alloy",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.5},
            "crystal_structure": "fcc"
        },
        "simulation_goal": {
            "type": "melt_quench",
            "temperature_range": [300, 1000]
        }
    }
    with input_yaml_path.open("w") as f:
        yaml.dump(input_data, f)

    # WHEN the user executes the command `mlip-auto init input.yaml`
    # We use init command.

    # We need to run it in uat_work_dir
    with pytest.MonkeyPatch.context() as m:
        m.chdir(uat_work_dir)
        result = runner.invoke(app, ["init", "input.yaml"])

    # THEN the standard output should display a success message
    assert result.exit_code == 0
    assert "SUCCESS" in result.stdout

    # AND a directory named `AlCu_Alloy` should be created
    project_dir = uat_work_dir / "AlCu_Alloy"
    assert project_dir.exists()
    assert project_dir.is_dir()

    # AND a file `project.db` should exist
    db_path = project_dir / "AlCu_Alloy.db" # Based on heuristics in factory.py
    assert db_path.exists()

    # AND a file `mlip_auto.log` should exist
    log_path = project_dir / "mlip_auto.log"
    assert log_path.exists()


def test_uat_01_02_invalid_configuration(uat_work_dir: Path) -> None:
    """
    UAT-01-02: Invalid Configuration Handling
    Verify that the system provides error messages for invalid configuration.
    """
    # GIVEN a file named `bad_input.yaml` containing invalid composition
    bad_input_path = uat_work_dir / "bad_input.yaml"
    bad_data = {
        "project_name": "Bad_Alloy",
        "target_system": {
            "elements": ["Fe"],
            "composition": {"Fe": 0.5}, # Sum != 1.0
            "crystal_structure": "bcc"
        },
        "simulation_goal": {"type": "elastic"}
    }
    with bad_input_path.open("w") as f:
        yaml.dump(bad_data, f)

    # WHEN the user executes the command
    with pytest.MonkeyPatch.context() as m:
        m.chdir(uat_work_dir)
        result = runner.invoke(app, ["init", "bad_input.yaml"])

    # THEN the system should exit with a non-zero error code
    assert result.exit_code != 0

    # AND the error message should state "Composition" error
    # (Pydantic validation error)
    assert "Composition" in result.stdout or "Composition" in str(result.exception)

def test_uat_01_03_database_provenance(uat_work_dir: Path) -> None:
    """
    UAT-01-03: Database & Provenance Check
    Verify that the initialized database contains the configuration settings in its metadata.
    """
    # GIVEN that the project has been successfully initialized (Reuse logic from test 1)
    input_yaml_path = uat_work_dir / "input_prov.yaml"
    input_data = {
        "project_name": "Prov_Test",
        "target_system": {
            "elements": ["Au"],
            "composition": {"Au": 1.0},
            "crystal_structure": "fcc"
        },
        "simulation_goal": {"type": "elastic"}
    }
    with input_yaml_path.open("w") as f:
        yaml.dump(input_data, f)

    with pytest.MonkeyPatch.context() as m:
        m.chdir(uat_work_dir)
        result = runner.invoke(app, ["init", "input_prov.yaml"])

    assert result.exit_code == 0

    project_dir = uat_work_dir / "Prov_Test"
    db_path = project_dir / "Prov_Test.db"

    # WHEN a user connects to the database
    assert db_path.exists()
    import ase.db
    conn = ase.db.connect(db_path)
    # Force connection initialization to read metadata
    try:
        conn.count()
    except Exception:
        pass

    # THEN the meta dictionary should be non-empty and contain system_config
    metadata = conn.metadata
    assert "system_config" in metadata
    assert metadata["system_config"]["project_name"] == "Prov_Test"
